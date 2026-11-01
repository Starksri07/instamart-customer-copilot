
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
TEST_FILE = STATE_DIR / "test_checkpoint.json"

STATE_DIR.mkdir(exist_ok=True)

checkpoint = {
    "order_id": "ORD-107301",
    "step": 3,
    "status": "INTERRUPTED",
    "messages": [
        {
            "role": "system",
            "content": "You are an AGENT. Process refund requests safely."
        },
        {
            "role": "user",
            "content": "Please check refund eligibility for order ORD-107301."
        },
        {
            "role": "assistant",
            "content": "Order retrieved. Policy check completed."
        }
    ]
}

TEST_FILE.write_text(
    json.dumps(checkpoint, indent=2),
    encoding="utf-8"
)

restored = json.loads(TEST_FILE.read_text(encoding="utf-8"))

print("========== CHECKPOINT RECOVERY TEST ==========")
print("Order ID:", restored["order_id"])
print("Saved step:", restored["step"])
print("Session status:", restored["status"])
print("Messages restored:", len(restored["messages"]))
print("Recovery:", "PASSED" if restored["status"] == "INTERRUPTED" else "FAILED")
