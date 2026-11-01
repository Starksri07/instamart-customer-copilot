
import csv
import json
import time
from pathlib import Path

from plan_execute_agent import run_plan_execute

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports"
LOG_FILE = ROOT / "outputs" / "refund_agent_runs.jsonl"

SCENARIOS = {
    "ORD-101136": ("Cancelled order", "DENIED"),
    "ORD-108250": ("Delivery within 45 minutes", "DENIED"),
    "ORD-107301": ("High-value refund", "ESCALATE"),
    "ORD-999999": ("Invalid order", "NOT_FOUND"),
}

def read_react_results():
    grouped = {}

    if not LOG_FILE.exists():
        return grouped

    with LOG_FILE.open(encoding="utf-8") as file:
        for line in file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            order_id = row.get("order_id")
            if order_id not in SCENARIOS:
                continue

            grouped.setdefault(order_id, []).append(row)

    results = {}

    for order_id, rows in grouped.items():
        final_rows = [
            row for row in rows
            if row.get("tool") == "final_response"
        ]

        if not final_rows:
            continue

        final_text = final_rows[-1].get("result", "")

        if final_text.startswith("DENIED"):
            status = "DENIED"
        elif final_text.startswith("ESCALATE"):
            status = "ESCALATE"
        elif final_text.startswith("NOT_FOUND"):
            status = "NOT_FOUND"
        elif final_text.startswith("REFUND"):
            status = "REFUNDED"
        else:
            status = "UNKNOWN"

        # Count calls from the latest execution ending in a final response.
        final_index = rows.index(final_rows[-1])
        execution_rows = rows[:final_index + 1]

        # Find the last preceding step-1 entry for this order.
        starts = [
            i for i, item in enumerate(execution_rows)
            if item.get("step") == 1
        ]
        start = starts[-1] if starts else 0
        current_execution = execution_rows[start:final_index + 1]

        calls = sum(
            1 for item in current_execution
            if item.get("tool") not in ("final_response", "max_steps")
        )

        results[order_id] = {
            "status": status,
            "tool_calls": calls,
            "passed": status == SCENARIOS[order_id][1],
        }

    return results


def evaluate_plan_execute():
    results = []

    for order_id, (scenario, expected) in SCENARIOS.items():
        start = time.perf_counter()
        result = run_plan_execute(order_id)
        elapsed = time.perf_counter() - start

        results.append({
            "agent": "Plan-and-Execute",
            "scenario": scenario,
            "order_id": order_id,
            "expected_status": expected,
            "actual_status": result["status"],
            "tool_calls": result["tool_calls"],
            "execution_seconds": round(elapsed, 4),
            "passed": result["status"] == expected,
        })

    return results


if __name__ == "__main__":
    react = read_react_results()
    plan = evaluate_plan_execute()

    comparison = []

    for row in plan:
        order_id = row["order_id"]
        react_result = react.get(order_id)

        comparison.append(row)

        if react_result:
            scenario, expected = SCENARIOS[order_id]
            comparison.append({
                "agent": "ReAct",
                "scenario": scenario,
                "order_id": order_id,
                "expected_status": expected,
                "actual_status": react_result["status"],
                "tool_calls": react_result["tool_calls"],
                "execution_seconds": "",
                "passed": react_result["passed"],
            })

    output = OUTPUT_DIR / "agent_comparison.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "agent", "scenario", "order_id",
                "expected_status", "actual_status",
                "tool_calls", "execution_seconds", "passed"
            ]
        )
        writer.writeheader()
        writer.writerows(comparison)

    print("\n========== AGENT COMPARISON ==========")
    for row in comparison:
        print(
            row["agent"], "|",
            row["scenario"], "|",
            "Calls:", row["tool_calls"],
            "| Status:", row["actual_status"],
            "| Passed:", row["passed"]
        )

    print("\nComparison report:", output)
