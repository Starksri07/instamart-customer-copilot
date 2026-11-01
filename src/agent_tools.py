import csv
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart_v2.db"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

REFUND_LOG = OUTPUT_DIR / "refund_actions.csv"


def get_order(order_id):
    """Retrieve an order from SQLite."""

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute("""
            SELECT
                order_id,
                customer_id,
                product_id,
                quantity,
                order_ts,
                status,
                discount_pct,
                order_amount,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
        """, (order_id,)).fetchone()

        if row is None:
            return {
                "success": False,
                "error": "Order not found."
            }

        return {
            "success": True,
            "order": dict(row)
        }

    finally:
        conn.close()


def search_policy(question):
    """Retrieve relevant policy sentences."""

    from policy_retrieval import PolicyRetriever

    retriever = PolicyRetriever()
    results = retriever.retrieve(
        question,
        top_k=5,
        min_score=0.05
    )

    if not results:
        return {
            "success": False,
            "error": "No relevant policy found."
        }

    return {
        "success": True,
        "results": results
    }


def calculate_refund(order_id):
    """Calculate refund eligibility from the refund policy."""

    result = get_order(order_id)

    if not result["success"]:
        return result

    order = result["order"]

    if order["status"] != "completed":
        return {
            "success": True,
            "eligible": False,
            "refund_amount": 0,
            "reason": "Only completed orders qualify for refunds.",
            "order": order
        }

    delivery_minutes = order["delivery_minutes"]

    if delivery_minutes is None:
        return {
            "success": False,
            "error": "Delivery time is unavailable. Human review required."
        }

    if delivery_minutes <= 45:
        return {
            "success": True,
            "eligible": False,
            "refund_amount": 0,
            "reason": "Delivery was within 45 minutes.",
            "order": order
        }

    amount = float(order["order_amount"])

    if amount <= 0:
        return {
            "success": False,
            "error": "Invalid order amount. Human review required."
        }

    if amount > 1000:
        return {
            "success": True,
            "eligible": True,
            "refund_amount": round(amount, 2),
            "approval_required": True,
            "reason": "Refund exceeds INR 1000. Human approval required.",
            "order": order
        }

    return {
        "success": True,
        "eligible": True,
        "refund_amount": round(amount, 2),
        "approval_required": False,
        "reason": "Eligible for automatic refund.",
        "order": order
    }


def issue_refund(order_id, amount):
    """Simulate an approved refund and record it."""

    calculation = calculate_refund(order_id)

    if not calculation["success"]:
        return calculation

    if not calculation["eligible"]:
        return {
            "success": False,
            "error": calculation["reason"]
        }

    if calculation.get("approval_required"):
        return {
            "success": False,
            "error": "Human approval is required before issuing this refund."
        }

    approved_amount = calculation["refund_amount"]

    if round(float(amount), 2) != approved_amount:
        return {
            "success": False,
            "error": "Requested amount does not match the eligible refund."
        }

    order = calculation["order"]

    if REFUND_LOG.exists():
        with REFUND_LOG.open(
            "r", newline="", encoding="utf-8"
        ) as file:
            existing = list(csv.DictReader(file))

        if any(row["order_id"] == order_id for row in existing):
            return {
                "success": False,
                "error": "A refund action is already recorded for this order."
            }

    new_file = not REFUND_LOG.exists()

    with REFUND_LOG.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "order_id",
                "customer_id",
                "refund_amount",
                "status",
                "created_at"
            ]
        )

        if new_file:
            writer.writeheader()

        writer.writerow({
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "refund_amount": approved_amount,
            "status": "SIMULATED_APPROVED",
            "created_at": datetime.now().isoformat()
        })

    return {
        "success": True,
        "status": "SIMULATED_APPROVED",
        "order_id": order_id,
        "refund_amount": approved_amount,
        "message": "Refund action recorded in the simulation."
    }


def escalate_to_human(order_id, reason):
    """Record a human-review escalation."""

    escalation_file = OUTPUT_DIR / "human_escalations.csv"
    new_file = not escalation_file.exists()

    with escalation_file.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "order_id",
                "reason",
                "status",
                "created_at"
            ]
        )

        if new_file:
            writer.writeheader()

        writer.writerow({
            "order_id": order_id,
            "reason": reason,
            "status": "PENDING_HUMAN_REVIEW",
            "created_at": datetime.now().isoformat()
        })

    return {
        "success": True,
        "status": "PENDING_HUMAN_REVIEW",
        "order_id": order_id,
        "message": "Case recorded for human support review."
    }


if __name__ == "__main__":

    print("\n========== PART D: AGENT TOOLS TEST ==========")

    conn = sqlite3.connect(DB_PATH)

    sample = conn.execute("""
        SELECT order_id
        FROM orders
        WHERE status = 'completed'
        LIMIT 1
    """).fetchone()

    conn.close()

    if sample is None:
        print("No completed sample order found.")
    else:
        order_id = sample[0]

        print("\nSample order:", order_id)

        order_result = get_order(order_id)
        print("\nGET ORDER:")
        print(order_result)

        refund_result = calculate_refund(order_id)
        print("\nCALCULATE REFUND:")
        print(refund_result)

    print("\nAgent tools loaded successfully.")
