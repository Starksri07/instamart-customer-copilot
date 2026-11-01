
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from agent_tools import (
    get_order,
    search_policy,
    calculate_refund,
    issue_refund,
    escalate_to_human,
)


def create_plan(order_id):
    """Create a plan before executing any tools."""
    return [
        "Retrieve order details",
        "Retrieve relevant refund policy",
        "Calculate refund eligibility and amount",
        "Decide whether to deny, refund, or escalate",
    ]


def finish(order_id, status, message, results, tool_calls):
    print("\nEXECUTION RESULTS:")
    for name, result in results:
        print(f"{name}: {result}")

    print(f"\nFINAL STATUS: {status}")
    print(f"MESSAGE: {message}")
    print(f"TOOL CALLS: {tool_calls}")

    return {
        "order_id": order_id,
        "status": status,
        "message": message,
        "tool_calls": tool_calls,
        "results": results,
    }


def run_plan_execute(order_id):
    print("\n========== PLAN-AND-EXECUTE AGENT ==========")

    plan = create_plan(order_id)

    print("\nEXECUTION PLAN:")
    for number, step in enumerate(plan, start=1):
        print(f"{number}. {step}")

    results = []
    tool_calls = 0

    # Step 1: Retrieve order.
    order_result = get_order(order_id)
    tool_calls += 1
    results.append(("get_order", order_result))

    if not order_result.get("success"):
        return finish(
            order_id, "NOT_FOUND", "Order not found.",
            results, tool_calls
        )

    # Step 2: Retrieve policy.
    policy_result = search_policy("refund policy late delivery")
    tool_calls += 1
    results.append(("search_policy", policy_result))

    if not policy_result.get("success"):
        return finish(
            order_id, "REVIEW",
            "Could not retrieve refund policy. Human review required.",
            results, tool_calls
        )

    # Step 3: Calculate eligibility.
    refund_result = calculate_refund(order_id)
    tool_calls += 1
    results.append(("calculate_refund", refund_result))

    if not refund_result.get("success"):
        return finish(
            order_id, "REVIEW",
            refund_result.get("error", "Human review required."),
            results, tool_calls
        )

    if not refund_result.get("eligible"):
        return finish(
            order_id, "DENIED",
            refund_result.get("reason", "Refund is not eligible."),
            results, tool_calls
        )

    amount = refund_result["refund_amount"]

    # Step 4: Decide whether human approval is required.
    if refund_result.get("approval_required"):
        escalation = escalate_to_human(
            order_id,
            f"Refund of INR {amount} requires human approval."
        )
        tool_calls += 1
        results.append(("escalate_to_human", escalation))

        return finish(
            order_id, "ESCALATE",
            "Refund requires human approval.",
            results, tool_calls
        )

    # Issue only an eligible simulated refund.
    refund_action = issue_refund(order_id, amount)
    tool_calls += 1
    results.append(("issue_refund", refund_action))

    if not refund_action.get("success"):
        return finish(
            order_id, "FAILED",
            refund_action.get("error", "Refund could not be issued."),
            results, tool_calls
        )

    return finish(
        order_id, "REFUNDED",
        f"Simulated refund of INR {amount} recorded.",
        results, tool_calls
    )


if __name__ == "__main__":
    order_id = input("Enter order ID for Plan-and-Execute test: ").strip()

    if order_id:
        run_plan_execute(order_id)
