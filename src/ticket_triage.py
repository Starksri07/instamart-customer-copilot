
from pathlib import Path
import sys
import re
import json
import time

import pandas as pd

# Import the original supplied mock LLM without modifying it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

TICKET_PATH = ROOT / "data" / "clean" / "tickets_clean_v2.csv"
SEGMENT_PATH = ROOT / "data" / "clean" / "customer_segments.csv"

OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "ticket_triage_results.csv"
EVALUATION_PATH = REPORT_DIR / "ticket_triage_evaluation.csv"
ROUTING_PATH = REPORT_DIR / "ticket_segment_routing.csv"

ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}

TOKEN_BUDGET = 1200
MAX_RETRIES = 4

# Prompt variants
PROMPT_BASIC = """
You are a customer support ticket classifier.

CLASSIFY each customer ticket into exactly one category:
billing, delivery, technical, or fraud.

Return ONLY valid JSON:
{"category": "category_name", "reason": "short reason"}
"""

PROMPT_EXAMPLES = """
You are a customer support ticket classifier.

CLASSIFY each customer ticket into exactly one category:
billing, delivery, technical, or fraud.

EXAMPLES:
- Unauthorized transaction: fraud
- App login failure: technical
- Incorrect payment deduction: billing
- Late parcel: delivery

Return ONLY valid JSON:
{"category": "category_name", "reason": "short reason"}
"""


# --------------------------------------------------
# 2. PRIVACY REDACTION
# --------------------------------------------------

def redact_pii(text):
    """Mask common personal identifiers before sending text to the LLM."""

    text = str(text)

    # Email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        text
    )

    # Indian mobile numbers and common international formats
    text = re.sub(
        r"(?<!\w)(?:\+91[\s-]?)?[6-9]\d{9}(?!\w)",
        "[PHONE]",
        text
    )

    # Order IDs
    text = re.sub(
        r"\bORD-\d{6}\b",
        "[ORDER_ID]",
        text,
        flags=re.IGNORECASE
    )

    return text


# --------------------------------------------------
# 3. TOKEN ESTIMATION
# --------------------------------------------------

def count_tokens(messages):
    total = 0

    for message in messages:
        total += mock_llm.est_tokens(message.get("content", ""))

    return total


# --------------------------------------------------
# 4. JSON RESPONSE VALIDATION
# --------------------------------------------------

def validate_response(content):
    if not content:
        return None

    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(result, dict):
        return None

    category = str(result.get("category", "")).lower().strip()

    if category not in ALLOWED_CATEGORIES:
        return None

    reason = str(result.get("reason", "")).strip()

    return {
        "category": category,
        "reason": reason
    }


# --------------------------------------------------
# 5. CLASSIFICATION WITH RETRIES
# --------------------------------------------------

def classify_ticket(ticket_text, prompt):
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": ticket_text
        }
    ]

    token_count = count_tokens(messages)

    if token_count > TOKEN_BUDGET:
        return {
            "category": "manual_review",
            "reason": "Token budget exceeded",
            "attempts": 0,
            "tokens": token_count,
            "status": "token_limit"
        }

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = mock_llm.chat(messages)

            content = response.choices[0].message.content

            result = validate_response(content)

            if result is not None:
                return {
                    "category": result["category"],
                    "reason": result["reason"],
                    "attempts": attempt,
                    "tokens": response.usage.total_tokens,
                    "status": "success"
                }

            # Invalid JSON or category: retry with stricter instruction.
            messages.append({
                "role": "assistant",
                "content": content or ""
            })

            messages.append({
                "role": "user",
                "content": (
                    "Your previous answer was invalid. "
                    "Return only valid JSON with category exactly "
                    "billing, delivery, technical, or fraud. "
                    "Required format: "
                    '{"category":"billing","reason":"short reason"}'
                )
            })

            token_count = count_tokens(messages)

            if token_count > TOKEN_BUDGET:
                return {
                    "category": "manual_review",
                    "reason": "Retry exceeded token budget",
                    "attempts": attempt,
                    "tokens": token_count,
                    "status": "token_limit"
                }

        except mock_llm.TransientLLMError:
            # Simulated temporary failure from the supplied mock LLM.
            if attempt == MAX_RETRIES:
                return {
                    "category": "manual_review",
                    "reason": "Temporary LLM failure after retries",
                    "attempts": attempt,
                    "tokens": token_count,
                    "status": "retry_failed"
                }

        except Exception as error:
            return {
                "category": "manual_review",
                "reason": f"Unexpected error: {type(error).__name__}",
                "attempts": attempt,
                "tokens": token_count,
                "status": "error"
            }

    return {
        "category": "manual_review",
        "reason": "Could not obtain a valid classification",
        "attempts": MAX_RETRIES,
        "tokens": token_count,
        "status": "invalid_response"
    }


# --------------------------------------------------
# 6. CUSTOMER SEGMENT ROUTING
# --------------------------------------------------

def load_customer_segments():

    if not SEGMENT_PATH.exists():
        print("\nSegment file not found. Using default routing.")
        return {}

    segments = pd.read_csv(SEGMENT_PATH)

    if "customer_id" not in segments.columns:
        print("No customer_id column in segment file.")
        return {}

    possible_columns = [
        "segment",
        "customer_segment",
        "segment_name",
        "cluster",
        "customer_cluster"
    ]

    segment_column = next(
        (column for column in possible_columns
         if column in segments.columns),
        None
    )

    if segment_column is None:
        print("No recognized segment column. Using default routing.")
        return {}

    return dict(
        zip(
            segments["customer_id"].astype(str),
            segments[segment_column].astype(str)
        )
    )


def route_ticket(category, customer_segment):

    segment = str(customer_segment).lower()

    if category == "fraud":
        return "URGENT_FRAUD_TEAM"

    if category == "manual_review":
        return "MANUAL_REVIEW"

    if category == "technical":
        return "TECHNICAL_SUPPORT"

    if category == "delivery":
        return "DELIVERY_SUPPORT"

    if category == "billing":
        return "BILLING_SUPPORT"

    # Priority routing for valuable customers.
    if any(word in segment for word in ["high", "vip", "premium"]):
        return "PRIORITY_CUSTOMER_SUPPORT"

    return "GENERAL_SUPPORT"


# --------------------------------------------------
# 7. MAIN TICKET PROCESSING
# --------------------------------------------------

def main():

    print("\nLoading ticket data...")

    if not TICKET_PATH.exists():
        raise FileNotFoundError(
            f"Ticket file not found: {TICKET_PATH}"
        )

    tickets = pd.read_csv(TICKET_PATH)

    required_columns = {
        "ticket_id",
        "customer_id",
        "text"
    }

    missing = required_columns - set(tickets.columns)

    if missing:
        raise ValueError(
            f"Missing required ticket columns: {sorted(missing)}"
        )

    segment_map = load_customer_segments()

    mock_llm.reset()

    results = []

    print("Total tickets:", len(tickets))
    print("\nStarting ticket classification...")

    for index, row in tickets.iterrows():

        original_text = str(row["text"])

        masked_text = redact_pii(original_text)

        # Alternate prompt variants to compare prompt styles.
        if index % 2 == 0:
            selected_prompt = PROMPT_BASIC
            prompt_variant = "basic"
        else:
            selected_prompt = PROMPT_EXAMPLES
            prompt_variant = "examples"

        prediction = classify_ticket(
            masked_text,
            selected_prompt
        )

        customer_id = str(row["customer_id"])

        customer_segment = segment_map.get(
            customer_id,
            "unknown"
        )

        assigned_team = route_ticket(
            prediction["category"],
            customer_segment
        )

        true_category = (
            str(row["true_category"]).lower().strip()
            if "true_category" in tickets.columns
            and pd.notna(row.get("true_category"))
            else ""
        )

        results.append({
            "ticket_id": row["ticket_id"],
            "customer_id": customer_id,
            "masked_text": masked_text,
            "true_category": true_category,
            "predicted_category": prediction["category"],
            "reason": prediction["reason"],
            "customer_segment": customer_segment,
            "assigned_team": assigned_team,
            "prompt_variant": prompt_variant,
            "attempts": prediction["attempts"],
            "estimated_tokens": prediction["tokens"],
            "status": prediction["status"]
        })

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(tickets)} tickets"
            )

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # 8. SAVE RESULTS
    # --------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    routing_summary = (
        results_df.groupby(
            ["predicted_category", "assigned_team"]
        )
        .size()
        .reset_index(name="ticket_count")
    )

    routing_summary.to_csv(
        ROUTING_PATH,
        index=False
    )

    # --------------------------------------------------
    # 9. EVALUATION
    # --------------------------------------------------

    valid = results_df[
        results_df["true_category"].isin(ALLOWED_CATEGORIES)
        & results_df["predicted_category"].isin(ALLOWED_CATEGORIES)
    ].copy()

    evaluation_rows = []

    for category in sorted(ALLOWED_CATEGORIES):

        actual = valid["true_category"] == category
        predicted = valid["predicted_category"] == category

        tp = int((actual & predicted).sum())
        fp = int((~actual & predicted).sum())
        fn = int((actual & ~predicted).sum())

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        evaluation_rows.append({
            "category": category,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    evaluation_df.to_csv(
        EVALUATION_PATH,
        index=False
    )

    # --------------------------------------------------
    # 10. FINAL SUMMARY
    # --------------------------------------------------

    print("\n========== PART C SUMMARY ==========")

    print("Total tickets:", len(results_df))

    print("\nClassification counts:")
    print(results_df["predicted_category"].value_counts())

    print("\nProcessing status:")
    print(results_df["status"].value_counts())

    print("\nAverage attempts:")
    print(round(results_df["attempts"].mean(), 2))

    print("\nEvaluation:")
    print(evaluation_df.to_string(index=False))

    print("\nSaved files:")
    print(OUTPUT_PATH)
    print(EVALUATION_PATH)
    print(ROUTING_PATH)

    print("\nPART C TICKET TRIAGE COMPLETED.")


if __name__ == "__main__":
    main()
