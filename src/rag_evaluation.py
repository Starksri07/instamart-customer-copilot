from pathlib import Path
import csv

from policy_retrieval import PolicyRetriever


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

retriever = PolicyRetriever()

test_cases = [
    {
        "question": "When can a customer receive a full refund?",
        "expected_source": "refund_policy.txt"
    },
    {
        "question": "How long does a failed payment refund take?",
        "expected_source": "payment_policy.txt"
    },
    {
        "question": "What happens when a customer reports fraud?",
        "expected_source": "escalation_policy.txt"
    },
    {
        "question": "How are customer phone numbers protected?",
        "expected_source": "privacy_policy.txt"
    },
    {
        "question": "How long is the standard delivery time?",
        "expected_source": "delivery_policy.txt"
    },
    {
        "question": "Explain quantum teleportation and distant galaxies.",
        "expected_source": "ZERO_OVERLAP"
    }
]

results = []

print("\n========== PART D: RAG EVALUATION ==========")

for number, case in enumerate(test_cases, start=1):

    question = case["question"]
    expected = case["expected_source"]

    retrieved = retriever.retrieve(
        question,
        top_k=3,
        min_score=0.05
    )

    sources = [item["source"] for item in retrieved]

    if expected == "ZERO_OVERLAP":
        passed = len(retrieved) == 0
        status = "PASS" if passed else "FAIL"

    else:
        passed = expected in sources
        status = "PASS" if passed else "FAIL"

    results.append({
        "test_id": number,
        "question": question,
        "expected_source": expected,
        "retrieved_sources": "; ".join(sources),
        "retrieved_count": len(retrieved),
        "passed": passed,
        "status": status
    })

    print(f"\nTest {number}: {status}")
    print("Question:", question)
    print("Expected:", expected)

    if retrieved:
        for item in retrieved:
            print(
                f"  {item['source']} "
                f"(score={item['score']})"
            )
    else:
        print("  No relevant policy information found.")

total = len(results)
passed_count = sum(row["passed"] for row in results)
accuracy = passed_count / total * 100

output_file = OUTPUT_DIR / "rag_evaluation.csv"

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print("\n========== EVALUATION SUMMARY ==========")
print("Total test cases:", total)
print("Passed:", passed_count)
print("Failed:", total - passed_count)
print(f"Evaluation pass rate: {accuracy:.2f}%")
print("Zero-overlap test:", results[-1]["status"])
print("Saved:", output_file)
