
import json
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart_v2.db"
STATE_DIR = ROOT / "state"
CHECKPOINT = STATE_DIR / "agent_checkpoint.json"
BACKUP = STATE_DIR / "agent_checkpoint_before_resume_test.json"

# Preserve the existing checkpoint.
if CHECKPOINT.exists() and not BACKUP.exists():
    shutil.copy2(CHECKPOINT, BACKUP)

# Use a cancelled order to prevent any refund.
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

row = conn.execute(
    "SELECT * FROM orders WHERE order_id = ?",
    ("ORD-101136",)
).fetchone()

conn.close()

if row is None:
    raise SystemExit("Test order not found.")

order = dict(row)

messages = [
    {
        "role": "system",
        "content": (
            "You are an AGENT. Use tools to check orders "
            "and process refund requests safely."
        )
    },
    {
        "role": "user",
        "content": (
            "Please check refund eligibility for order ORD-101136."
        )
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "get_order",
        "content": json.dumps(order)
    }
]

checkpoint = {
    "order_id": "ORD-101136",
    "step": 1,
    "status": "RUNNING",
    "messages": messages
}

CHECKPOINT.write_text(
    json.dumps(checkpoint, indent=2),
    encoding="utf-8"
)

print("Test checkpoint prepared.")
print("Order:", order["order_id"])
print("Status:", order["status"])
print("Saved step:", checkpoint["step"])
print("Checkpoint status:", checkpoint["status"])
print("Backup:", BACKUP.name)
