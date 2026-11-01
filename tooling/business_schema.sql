PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS business_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    app_user_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    email TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    price_cents INTEGER NOT NULL CHECK(price_cents >= 0)
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    status TEXT NOT NULL CHECK(status IN ('delivered', 'shipped', 'completed')),
    return_status TEXT NOT NULL DEFAULT 'none' CHECK(return_status IN ('none', 'requested')),
    payment_method TEXT NOT NULL,
    purchased_at TEXT NOT NULL,
    delivered_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS order_items (
    item_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_paid_cents INTEGER NOT NULL CHECK(unit_paid_cents >= 0),
    returned_quantity INTEGER NOT NULL DEFAULT 0 CHECK(returned_quantity >= 0 AND returned_quantity <= quantity),
    UNIQUE(order_id, item_id)
);
CREATE TABLE IF NOT EXISTS return_preparations (
    preparation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES customers(app_user_id),
    conv_id TEXT NOT NULL,
    prepared_request_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_paid_cents INTEGER NOT NULL CHECK(unit_paid_cents >= 0),
    refund_amount_cents INTEGER NOT NULL CHECK(refund_amount_cents >= 0),
    reason TEXT NOT NULL,
    reason_code TEXT NOT NULL CHECK(reason_code IN ('unwanted', 'quality_issue', 'other')),
    status TEXT NOT NULL CHECK(status IN ('prepared', 'submitted', 'superseded')),
    display_text TEXT NOT NULL,
    confirmation_text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY(order_id, item_id) REFERENCES order_items(order_id, item_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_prepared_item_per_chat
    ON return_preparations(user_id, conv_id, item_id) WHERE status = 'prepared';
CREATE TABLE IF NOT EXISTS returns (
    return_id TEXT PRIMARY KEY,
    preparation_id TEXT NOT NULL UNIQUE REFERENCES return_preparations(preparation_id),
    user_id TEXT NOT NULL REFERENCES customers(app_user_id),
    conv_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    refund_amount_cents INTEGER NOT NULL CHECK(refund_amount_cents >= 0),
    reason TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status = 'submitted'),
    refund_status TEXT NOT NULL CHECK(refund_status = 'not_issued'),
    submitted_request_id TEXT NOT NULL,
    confirmation_message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(order_id, item_id) REFERENCES order_items(order_id, item_id)
);
CREATE TABLE IF NOT EXISTS return_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    return_id TEXT NOT NULL REFERENCES returns(return_id),
    event_type TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(return_id, event_type)
);
CREATE TABLE IF NOT EXISTS handoffs (
    handoff_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES customers(app_user_id),
    conv_id TEXT NOT NULL,
    order_id TEXT REFERENCES orders(order_id),
    reason TEXT NOT NULL,
    summary TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status = 'recorded'),
    dedupe_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS business_operations (
    operation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    conv_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    server_pid INTEGER NOT NULL,
    operation TEXT NOT NULL CHECK(operation IN ('read', 'write')),
    rows_changed INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS orders_by_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS returns_by_user ON returns(user_id, order_id);
CREATE INDEX IF NOT EXISTS preparations_by_chat ON return_preparations(user_id, conv_id);
