from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policies"

POLICY_TITLES = {
    "Delivery Policy.",
    "Escalation Policy.",
    "Payment Policy.",
    "Privacy Policy.",
    "Refund Policy."
}


def load_policy_documents():
    documents = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence and sentence not in POLICY_TITLES:
                documents.append({
                    "source": file_path.name,
                    "text": sentence
                })

    return documents


class PolicyRetriever:

    def __init__(self):
        self.documents = load_policy_documents()

        if not self.documents:
            raise ValueError("No policy documents found.")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.document_matrix = self.vectorizer.fit_transform(
            [doc["text"] for doc in self.documents]
        )

    def retrieve(self, question, top_k=3, min_score=0.05):

        if not question.strip():
            return []

        query_vector = self.vectorizer.transform([question])

        scores = cosine_similarity(
            query_vector,
            self.document_matrix
        ).flatten()

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < min_score:
                break

            results.append({
                "source": self.documents[index]["source"],
                "text": self.documents[index]["text"],
                "score": round(score, 4)
            })

            if len(results) >= top_k:
                break

        return results


def main():

    print("\n========== PART D: IMPROVED POLICY RETRIEVAL ==========")

    retriever = PolicyRetriever()

    print("Policy documents:", len(set(
        doc["source"] for doc in retriever.documents
    )))

    print("Policy sentences:", len(retriever.documents))

    questions = [
        "When can a customer receive a full refund?",
        "How long does a failed payment refund take?",
        "What happens when a customer reports fraud?",
        "How are customer phone numbers protected?"
    ]

    for question in questions:

        print("\nQuestion:", question)

        results = retriever.retrieve(question)

        if not results:
            print("No relevant policy information found.")
            continue

        for result in results:
            print(
                f"\nSource: {result['source']} | "
                f"Score: {result['score']}"
            )
            print("Policy:", result["text"])


if __name__ == "__main__":
    main()
