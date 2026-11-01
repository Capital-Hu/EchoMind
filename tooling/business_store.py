"""由 MCP Server 持有的业务 SQLite；固定业务操作、事务、归属校验和提交去重。"""
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = Path(__file__).with_name("business_schema.sql")
SEED = ROOT / "fixtures" / "business_demo.json"
WRITE_TOOLS = {"return_prepare", "return_submit", "handoff_create"}


def now():
    return datetime.now(timezone.utc)


def money(cents):
    return f"{cents // 100}.{cents % 100:02d}"


def normalize_display(text):
    return re.sub(r"[\s*`]+", "", text)


class BusinessStore:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("BUSINESS_DB_PATH", str(ROOT / "data/business/business.sqlite3"))).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA.read_text(encoding="utf-8"))
        self._seed()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _transaction(self):
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def _seed(self):
        raw = SEED.read_bytes()
        seed = json.loads(raw)
        with self._transaction() as db:
            meta = dict(db.execute("SELECT key,value FROM business_meta").fetchall())
            if meta:
                if meta.get("schema_version") != "1":
                    raise RuntimeError("业务库版本与当前代码不一致")
                return
            if db.execute("SELECT COUNT(*) FROM customers").fetchone()[0]:
                raise RuntimeError("业务库缺少来源记录，拒绝覆盖已有客户数据")
            timestamp = now()
            for customer in seed["customers"]:
                db.execute("INSERT INTO customers VALUES(?,?,?,?)", tuple(customer[key] for key in (
                    "customer_id", "app_user_id", "name", "email")))
            for product in seed["products"]:
                db.execute("INSERT INTO products VALUES(?,?,?,?)", tuple(product[key] for key in (
                    "product_id", "name", "description", "price_cents")))
            for order in seed["orders"]:
                age = order["days_since_delivery"]
                delivered = (timestamp - timedelta(days=age)).isoformat() if age is not None else None
                purchased = (timestamp - timedelta(days=(age or 0) + 2)).isoformat()
                db.execute("INSERT INTO orders VALUES(?,?,?,'none','original_payment',?,?,?)",
                           (order["order_id"], order["customer_id"], order["status"], purchased, delivered, timestamp.isoformat()))
                for item in order["items"]:
                    db.execute("INSERT INTO order_items VALUES(?,?,?,?,?,0)", (
                        item["item_id"], order["order_id"], item["product_id"], item["quantity"], item["unit_paid_cents"]))
            values = {"schema_version": "1", "seed_version": seed["version"], "source": seed["source"],
                      "seed_sha256": hashlib.sha256(raw).hexdigest(), "seeded_at": timestamp.isoformat()}
            db.executemany("INSERT INTO business_meta VALUES(?,?)", values.items())

    async def call(self, name, arguments, context):
        return await asyncio.to_thread(self._call, name, arguments, context)

    def _call(self, name, arguments, context):
        from tooling.mcp_protocol import CATALOG, validate_arguments
        validate_arguments(CATALOG[name].schema, arguments)
        handlers = {"customer_lookup": self._customer_lookup, "order_list": self._order_list,
                    "order_get": self._order_get, "product_get": self._product_get,
                    "return_prepare": self._return_prepare, "return_submit": self._return_submit,
                    "return_status": self._return_status, "handoff_create": self._handoff_create,
                    "handoff_status": self._handoff_status, "business_stats": self._stats}
        with self._transaction() as db:
            if name != "business_stats":
                self._customer(db, context)
            before = db.total_changes
            data = handlers[name](db, arguments, context)
            changed = db.total_changes - before
            operation = "write" if name in WRITE_TOOLS else "read"
            cursor = db.execute("""INSERT INTO business_operations
                (tool_name,user_id,conv_id,request_id,call_id,server_pid,operation,rows_changed,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""", (name, context["user_id"], context["conv_id"], context["request_id"],
                context["call_id"], os.getpid(), operation, changed, now().isoformat()))
            data["database"] = {"path": str(self.path), "operation_id": cursor.lastrowid,
                                "operation": operation, "rows_changed": changed, "server_pid": os.getpid()}
            return data

    @staticmethod
    def _customer(db, context):
        customer = db.execute("SELECT * FROM customers WHERE app_user_id=?", (context["user_id"],)).fetchone()
        if customer is None:
            raise ValueError("当前用户未关联业务客户；演示账号为 u1001、u1002，请先选择对应账号")
        return dict(customer)

    def _customer_lookup(self, db, args, context):
        return {"customer": self._customer(db, context), "identity_source": "backend_user_id", "demo_data": True}

    @staticmethod
    def _owned_order(db, order_id, context):
        order = db.execute("""SELECT o.* FROM orders o JOIN customers c ON c.customer_id=o.customer_id
            WHERE o.order_id=? AND c.app_user_id=?""", (order_id, context["user_id"])).fetchone()
        if order is None:
            raise ValueError("未找到当前用户的该订单")
        return dict(order)

    @staticmethod
    def _items(db, order_id):
        return [dict(row) for row in db.execute("""SELECT i.*,p.name AS product_name,
            i.quantity-i.returned_quantity AS remaining_quantity FROM order_items i
            JOIN products p ON p.product_id=i.product_id WHERE i.order_id=? ORDER BY i.item_id""", (order_id,))]

    def _order_list(self, db, args, context):
        rows = db.execute("""SELECT o.* FROM orders o JOIN customers c ON c.customer_id=o.customer_id
            WHERE c.app_user_id=? ORDER BY o.order_id LIMIT 50""", (context["user_id"],))
        return {"orders": [{**dict(row), "items": self._items(db, row["order_id"])} for row in rows]}

    def _order_get(self, db, args, context):
        order = self._owned_order(db, args["order_id"], context)
        return {"order": {**order, "items": self._items(db, order["order_id"])}}

    @staticmethod
    def _product_get(db, args, context):
        product = db.execute("SELECT * FROM products WHERE product_id=?", (args["product_id"],)).fetchone()
        if product is None:
            raise ValueError("商品不存在")
        return {"product": dict(product)}

    def _eligible_item(self, db, args, context):
        order = self._owned_order(db, args["order_id"], context)
        if order["status"] not in {"delivered", "completed"} or not order["delivered_at"]:
            raise ValueError("订单尚未签收，不能准备退货；可记录交接申请")
        age = now() - datetime.fromisoformat(order["purchased_at"])
        if age < timedelta(0) or age > timedelta(days=7):
            raise ValueError("订单不在购买后7天的自助退货期内；质量问题需要人工核实，请使用交接工具")
        item = next((row for row in self._items(db, order["order_id"]) if row["item_id"] == args["item_id"]), None)
        if item is None:
            raise ValueError("该商品明细不属于当前订单")
        if args["quantity"] > item["remaining_quantity"]:
            raise ValueError("退货数量超过当前可退数量，可能已经提交过退货，请查询进度")
        return order, item

    @staticmethod
    def _preparation_result(row, replayed=False):
        return {"preparation": dict(row), "replayed": replayed, "submitted": False,
                "next_step": "请完整展示 display_text，等待用户下一轮确认，再调用 return_submit"}

    def _return_prepare(self, db, args, context):
        args = {"quantity": 1, "reason_code": "unwanted", **args}
        order, item = self._eligible_item(db, args, context)
        previous = db.execute("""SELECT * FROM return_preparations WHERE user_id=? AND conv_id=?
            AND item_id=? AND status='prepared'""", (context["user_id"], context["conv_id"], args["item_id"])).fetchone()
        if previous and previous["expires_at"] > now().isoformat() and all(previous[key] == args[key] for key in (
                "order_id", "quantity", "reason", "reason_code")) and previous["unit_paid_cents"] == item["unit_paid_cents"]:
            return self._preparation_result(previous, replayed=True)
        if previous:
            db.execute("UPDATE return_preparations SET status='superseded' WHERE preparation_id=?", (previous["preparation_id"],))
        preparation_id = "PREP-" + uuid.uuid4().hex[:12].upper()
        amount = item["unit_paid_cents"] * args["quantity"]
        phrase = f"确认退货 {preparation_id}"
        display = (f"订单 {order['order_id']}；商品 {item['product_name']}；数量 {args['quantity']}；"
                   f"预计退款 {money(amount)} 元，退回原支付渠道；原因 {args['reason']}。"
                   f"这里只准备了退货申请，尚未退款。确认请回复：{phrase}")
        timestamp = now()
        db.execute("""INSERT INTO return_preparations VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            preparation_id, context["user_id"], context["conv_id"], context["request_id"], order["order_id"],
            args["item_id"], args["quantity"], item["unit_paid_cents"], amount, args["reason"], args["reason_code"],
            "prepared", display, phrase, timestamp.isoformat(), (timestamp + timedelta(minutes=30)).isoformat()))
        row = db.execute("SELECT * FROM return_preparations WHERE preparation_id=?", (preparation_id,)).fetchone()
        return self._preparation_result(row)

    @staticmethod
    def _return_result(db, row, replayed=False):
        events = [dict(event) for event in db.execute("SELECT * FROM return_events WHERE return_id=? ORDER BY event_id", (row["return_id"],))]
        public = {key: value for key, value in dict(row).items() if key != "confirmation_message"}
        return {"return": public, "events": events, "replayed": replayed,
                "message": "退货申请已写入业务库，等待后续退货审核和收货；未发起支付退款"}

    def _return_submit(self, db, args, context):
        prep = db.execute("""SELECT * FROM return_preparations WHERE preparation_id=? AND user_id=? AND conv_id=?""",
                          (args["preparation_id"], context["user_id"], context["conv_id"])).fetchone()
        if prep is None:
            raise ValueError("未找到当前用户当前会话的待确认退货")
        existing = db.execute("SELECT * FROM returns WHERE preparation_id=?", (prep["preparation_id"],)).fetchone()
        if existing:
            return self._return_result(db, existing, replayed=True)
        if prep["status"] != "prepared" or prep["expires_at"] <= now().isoformat():
            raise ValueError("退货准备已失效，请重新准备并确认")
        if prep["prepared_request_id"] == context["request_id"]:
            raise ValueError("不能在准备明细的同一轮提交，请展示明细并等待用户下一轮确认")
        if normalize_display(prep["display_text"]) not in normalize_display(context.get("previous_reply", "")):
            raise ValueError("上一轮未完整展示这份退货明细，请原样展示 display_text 后再等待用户确认")
        message = context.get("user_message", "").strip().rstrip("。.!！")
        generic = message in {"确认", "确认提交", "确认退货", "同意提交", "确认，提交", "是的，确认提交"}
        pending = db.execute("""SELECT COUNT(*) FROM return_preparations WHERE user_id=? AND conv_id=?
            AND status='prepared' AND expires_at>?""", (context["user_id"], context["conv_id"], now().isoformat())).fetchone()[0]
        if message != prep["confirmation_text"] and not (generic and pending == 1):
            raise ValueError("当前用户原话没有明确确认这份退货，请用户回复 confirmation_text；模型不能代为确认")
        _, item = self._eligible_item(db, dict(prep), context)
        if item["unit_paid_cents"] != prep["unit_paid_cents"]:
            raise ValueError("订单实付金额已变化，请重新准备明细并确认")
        return_id = "RET-" + uuid.uuid4().hex[:12].upper()
        timestamp = now().isoformat()
        db.execute("INSERT INTO returns VALUES(?,?,?,?,?,?,?,?,?,?,'submitted','not_issued',?,?,?)", (
            return_id, prep["preparation_id"], context["user_id"], context["conv_id"], prep["order_id"],
            prep["item_id"], prep["quantity"], prep["refund_amount_cents"], prep["reason"], prep["reason_code"],
            context["request_id"], context["user_message"], timestamp))
        db.execute("UPDATE order_items SET returned_quantity=returned_quantity+? WHERE item_id=?", (prep["quantity"], prep["item_id"]))
        db.execute("UPDATE orders SET return_status='requested',updated_at=? WHERE order_id=?", (timestamp, prep["order_id"]))
        db.execute("UPDATE return_preparations SET status='submitted' WHERE preparation_id=?", (prep["preparation_id"],))
        db.execute("""INSERT INTO return_events(return_id,event_type,description,created_at) VALUES(?,'submitted',?,?)""",
                   (return_id, "用户确认，退货申请已提交；等待审核与退货，未发起支付退款", timestamp))
        row = db.execute("SELECT * FROM returns WHERE return_id=?", (return_id,)).fetchone()
        return self._return_result(db, row)

    def _return_status(self, db, args, context):
        if "order_id" in args:
            self._owned_order(db, args["order_id"], context)
        conditions, values = ["user_id=?"], [context["user_id"]]
        for key in ("order_id", "return_id"):
            if key in args:
                conditions.append(f"{key}=?")
                values.append(args[key])
        rows = db.execute("SELECT * FROM returns WHERE " + " AND ".join(conditions) + " ORDER BY created_at DESC LIMIT 20", values)
        returned = [self._return_result(db, row) for row in rows]
        preparations = db.execute("""SELECT * FROM return_preparations WHERE user_id=? AND conv_id=?
            AND status='prepared' AND expires_at>? ORDER BY created_at DESC LIMIT 20""",
            (context["user_id"], context["conv_id"], now().isoformat()))
        pending = [dict(row) for row in preparations if "return_id" not in args
                   and ("order_id" not in args or row["order_id"] == args["order_id"])]
        return {"returns": returned, "preparations": pending, "refund_issued": False}

    def _handoff_create(self, db, args, context):
        order_id = args.get("order_id")
        if order_id:
            self._owned_order(db, order_id, context)
        key = hashlib.sha256(json.dumps([context["user_id"], context["conv_id"], order_id]).encode()).hexdigest()
        previous = db.execute("SELECT * FROM handoffs WHERE dedupe_key=?", (key,)).fetchone()
        replayed = previous is not None
        if previous is None:
            handoff_id = "HO-" + uuid.uuid4().hex[:12].upper()
            db.execute("INSERT INTO handoffs VALUES(?,?,?,?,?,?,'recorded',?,?)", (
                handoff_id, context["user_id"], context["conv_id"], order_id, args["reason"], args["summary"], key, now().isoformat()))
            previous = db.execute("SELECT * FROM handoffs WHERE handoff_id=?", (handoff_id,)).fetchone()
        return {"handoff": dict(previous), "replayed": replayed, "human_connected": False,
                "summary_source": "assistant", "message": "交接申请已记录，尚未连接真实人工客服"}

    @staticmethod
    def _handoff_status(db, args, context):
        rows = db.execute("SELECT * FROM handoffs WHERE user_id=? AND conv_id=? ORDER BY created_at DESC LIMIT 20",
                          (context["user_id"], context["conv_id"]))
        return {"handoffs": [dict(row) for row in rows], "human_connected": False}

    def _stats(self, db, args, context):
        tables = ("customers", "products", "orders", "order_items", "return_preparations", "returns", "return_events", "handoffs")
        return {"path": str(self.path), "demo_data": True, "seed_file": str(SEED),
                "source": dict(db.execute("SELECT key,value FROM business_meta")),
                "counts": {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}}
