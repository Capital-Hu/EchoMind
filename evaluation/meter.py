"""Cross-process request budgets and provider-reported usage."""
import json
import sqlite3
import time


class MeteredClient:
    def __init__(self, client, path, role, limit=80):
        self.client, self.path, self.role, self.limit = client, str(path), role, limit
        self.messages = self
        with sqlite3.connect(self.path) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, role TEXT, status TEXT, elapsed_ms REAL, usage TEXT, request TEXT, response TEXT)''')

    async def create(self, **kwargs):
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM calls WHERE role=?", (self.role,)).fetchone()[0]
            if count >= self.limit:
                raise RuntimeError(f"{self.role} model request budget exhausted")
            call_id = db.execute("INSERT INTO calls(role,status,request) VALUES(?,'started',?)",
                                 (self.role, json.dumps(kwargs, ensure_ascii=False))).lastrowid
        started = time.monotonic()
        usage, status, response_data = None, "error", None
        try:
            result = await self.client.messages.create(**kwargs)
            usage = result.usage.model_dump()
            response_data = result.model_dump(mode="json")
            status = "completed"
            return result
        finally:
            with sqlite3.connect(self.path, timeout=30) as db:
                db.execute("UPDATE calls SET status=?,elapsed_ms=?,usage=?,response=? WHERE id=?",
                           (status, (time.monotonic()-started)*1000, json.dumps(usage), json.dumps(response_data, ensure_ascii=False), call_id))

    async def close(self):
        await self.client.close()


def usage_summary(path):
    if not path.exists():
        return {}
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT role,status,usage FROM calls").fetchall()
    result = {}
    for role, status, raw in rows:
        group = result.setdefault(role, {"calls": 0, "errors": 0, "missing_usage": 0,
                                         "input_tokens": 0, "output_tokens": 0})
        group["calls"] += 1
        group["errors"] += status != "completed"
        usage = json.loads(raw) if raw else None
        if usage is None:
            group["missing_usage"] += 1
        else:
            group["input_tokens"] += usage.get("input_tokens", 0)
            group["output_tokens"] += usage.get("output_tokens", 0)
    return result
