
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm
from src import agent_tools


STATE_DIR = ROOT / "state"
OUTPUT_DIR = ROOT / "outputs"

STATE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_FILE = STATE_DIR / "agent_checkpoint.json"
AGENT_LOG = OUTPUT_DIR / "refund_agent_runs.jsonl"

MAX_STEPS = 6


def save_checkpoint(messages, order_id, step, status):
    """Save the current agent state."""

    checkpoint = {
        "order_id": order_id,
        "step": step,
        "status": status,
        "messages": messages
    }

    CHECKPOINT_FILE.write_text(
        json.dumps(checkpoint, indent=2, default=str),
        encoding="utf-8"
    )


def load_checkpoint():
    """Load saved agent state."""

    if not CHECKPOINT_FILE.exists():
        return None

    return json.loads(
        CHECKPOINT_FILE.read_text(encoding="utf-8")
    )


def log_event(order_id, step, tool_name, result):
    """Record agent activity."""

    record = {
        "order_id": order_id,
        "step": step,
        "tool": tool_name,
        "result": result
    }

    with AGENT_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def execute_tool(name, arguments, current_order_id):
    """
    Convert mock LLM tool arguments into existing
    agent_tools function arguments.
    """

    order_id = arguments.get("order_id") or current_order_id

    if name == "get_order":

        result = agent_tools.get_order(order_id)

        if not result.get("success"):
            return {
                "error": result.get("error", "Order not found.")
            }

        # Mock LLM expects order fields at the top level.
        return result["order"]

    elif name == "search_policy":

        return agent_tools.search_policy(
            arguments.get("query", "refund policy")
        )

    elif name == "calculate_refund":

        # The mock LLM sends amount and delivery time,
        # but the actual tool requires order_id.
        result = agent_tools.calculate_refund(order_id)

        if not result.get("success"):
            return {
                "refund_inr": 0,
                "error": result.get(
                    "error",
                    "Refund calculation failed."
                )
            }

        return {
            "refund_inr": result.get("refund_amount", 0),
            "reason": result.get("reason", ""),
            "approval_required": result.get(
                "approval_required", False
            )
        }
    elif name == "issue_refund":

        refund_amount = arguments.get("refund_inr", 0)

        result = agent_tools.issue_refund(
            order_id,
            refund_amount
        )

        # Return the exact key expected by mock_llm.py
        if result.get("success"):
            return {
                "success": True,
                "order_id": order_id,
                "refund_inr": result.get("refund_amount", refund_amount),
                "status": result.get("status"),
                "message": result.get("message")
            }

        return {
            "success": False,
            "refund_inr": 0,
            "error": result.get("error", "Refund failed.")
        }

    elif name == "escalate_to_human":

        result = agent_tools.escalate_to_human(
            order_id,
            "Refund above INR 1000 requires human approval."
        )

        return result

    return {
        "success": False,
        "error": f"Unknown tool: {name}"
    }


def run_agent(user_message, resume=False):
    """Execute the ReAct refund agent."""

    if resume:

        checkpoint = load_checkpoint()

        if checkpoint is None:
            print("No checkpoint found.")
            return "NO_CHECKPOINT"

        if checkpoint.get("status") == "COMPLETED":
            print("This checkpoint is already completed.")
            return "ALREADY_COMPLETED"

        messages = checkpoint["messages"]
        order_id = checkpoint["order_id"]
        start_step = checkpoint["step"]

        print("Resuming order:", order_id)

    else:

        matches = re.findall(
            r"ORD-\d+",
            user_message.upper()
        )

        if matches and not re.fullmatch(
            r"ORD-\d{6}",
            matches[0]
        ):
            print(
                "Invalid order ID. Use ORD- followed by exactly 6 digits."
            )
            return "INVALID_ORDER_ID"

        order_id = matches[0] if matches else None

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
                "content": user_message
            }
        ]

        start_step = 0

    for step in range(start_step, MAX_STEPS):

        print(f"\n--- Agent Step {step + 1} ---")

        try:
            response = mock_llm.chat(messages, tools=None)

        except mock_llm.TransientLLMError:

            print("Temporary LLM error. Retrying once.")

            try:
                response = mock_llm.chat(messages, tools=None)

            except mock_llm.TransientLLMError:

                save_checkpoint(
                    messages,
                    order_id,
                    step,
                    "PAUSED_LLM_ERROR"
                )

                return "PAUSED_LLM_ERROR"

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # No tool call means the agent has finished.
        if not tool_calls:

            final_answer = message.content or "No response."

            print("\nAGENT RESULT:")
            print(final_answer)

            messages.append({
                "role": "assistant",
                "content": final_answer
            })

            save_checkpoint(
                messages,
                order_id,
                step + 1,
                "COMPLETED"
            )

            log_event(
                order_id,
                step + 1,
                "final_response",
                final_answer
            )

            return final_answer

        # Execute every requested tool.
        for tool_call in tool_calls:

            tool_name = tool_call.name
            arguments = tool_call.arguments

            print("Tool requested:", tool_name)
            print("Arguments:", arguments)

            try:
                result = execute_tool(
                    tool_name,
                    arguments,
                    order_id
                )

            except Exception as error:

                result = {
                    "success": False,
                    "error": str(error)
                }

            print("Tool result:", result)

            log_event(
                order_id,
                step + 1,
                tool_name,
                result
            )

            # Add result to the conversation in the format
            # expected by mock_llm.py.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(
                    result,
                    default=str
                )
            })

            save_checkpoint(
                messages,
                order_id,
                step + 1,
                "RUNNING"
            )

    # Maximum step limit reached.
    final_answer = (
        "ESCALATE: Maximum agent steps reached. "
        "Human review required."
    )

    print("\n", final_answer)

    save_checkpoint(
        messages,
        order_id,
        MAX_STEPS,
        "MAX_STEPS_REACHED"
    )

    log_event(
        order_id,
        MAX_STEPS,
        "max_steps",
        final_answer
    )

    return final_answer


if __name__ == "__main__":

    print("\n========== PART D: REACT REFUND AGENT ==========")

    print("\n1. New refund request")
    print("2. Resume checkpoint")

    choice = input("\nSelect option: ").strip()

    if choice == "2":

        run_agent("", resume=True)

    elif choice == "1":

        order_id = input(
            "\nEnter order ID (example ORD-110227): "
        ).strip().upper()

        if not re.fullmatch(r"ORD-\d{6}", order_id):

            print(
                "Invalid order ID. Enter exactly ORD- followed by 6 digits."
            )

        else:

            run_agent(
                f"Please check refund eligibility for order {order_id}."
            )

    else:

        print("Invalid selection.")
