import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm
from policy_retrieval import PolicyRetriever

retriever = PolicyRetriever()

FALLBACK = (
    "I could not find that information in the policy documents. "
    "Please contact a human support agent."
)


def answer_question(question):

    normalized = question.lower()

    # Distinguish failed-payment reversals from order refunds.
    if (
        "failed payment" in normalized
        or "money deducted" in normalized
        or "payment failed" in normalized
    ):
        search_query = (
            "payment fails money deducted amount returned "
            "automatically within 48 hours"
        )
        required_source = "payment_policy.txt"

    elif "fraud" in normalized or "unauthorized" in normalized:
        search_query = question
        required_source = "escalation_policy.txt"

    elif "phone number" in normalized or "email" in normalized:
        search_query = question
        required_source = "privacy_policy.txt"

    elif "delivery" in normalized or "delivered" in normalized:
        search_query = question
        required_source = "delivery_policy.txt"

    elif "refund" in normalized:
        search_query = question
        required_source = "refund_policy.txt"

    else:
        search_query = question
        required_source = None

    retrieved = retriever.retrieve(
        search_query,
        top_k=10,
        min_score=0.05
    )

    # Prefer the relevant policy document for known question types.
    if required_source:
        matching = [
            item for item in retrieved
            if item["source"] == required_source
        ]
        retrieved = matching[:1]

    else:
        retrieved = retrieved[:3]

    if not retrieved:
        return {
            "answer": FALLBACK,
            "sources": [],
            "status": "fallback"
        }

    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT\n"
                "Answer using only the supplied policy context. "
                "Do not combine different policy rules. "
                "Do not invent information."
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}"
            )
        }
    ]

    try:
        response = mock_llm.chat(messages)
        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            answer = FALLBACK
            status = "fallback"
        else:
            status = "answered"

    except mock_llm.TransientLLMError:
        answer = FALLBACK
        status = "fallback"

    return {
        "answer": answer,
        "sources": sorted(set(
            item["source"] for item in retrieved
        )),
        "status": status
    }


def main():

    print("\n========== PART D: RAG POLICY ASSISTANT ==========")

    questions = [
        "When can a customer receive a full refund?",
        "How long does a failed payment refund take?",
        "What happens when a customer reports fraud?",
        "How are customer phone numbers protected?",
        "Explain quantum teleportation and distant galaxies."
    ]

    for number, question in enumerate(questions, start=1):

        result = answer_question(question)

        print(f"\nTest {number}")
        print("Question:", question)
        print("Answer:", result["answer"])
        print("Sources:", result["sources"])
        print("Status:", result["status"])


if __name__ == "__main__":
    main()
