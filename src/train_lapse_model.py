
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

DATA_FILE = CLEAN / "lapse_training_data.csv"
MODEL_FILE = ROOT / "src" / "lapse_model.joblib"

df = pd.read_csv(DATA_FILE)

features = ["recency_days", "frequency", "monetary"]
target = "lapsed_60d"

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df[target].astype(int)

# Hold out 25% for evaluation and preserve the target proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        auc = float("nan")

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": auc,
    })

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    if pd.notna(auc):
        print(f"ROC-AUC: {auc:.3f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Returned", "Potentially Lapsed"],
        zero_division=0,
    ))

    print("Confusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    if name == "Random Forest":
        # Save the fitted pipeline and feature names together.
        joblib.dump(
            {"model": model, "features": features},
            MODEL_FILE,
        )

results_df = pd.DataFrame(results)
results_df.to_csv(REPORTS / "lapse_model_comparison.csv", index=False)

print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))
print(f"\nComparison saved: {REPORTS / 'lapse_model_comparison.csv'}")
print(f"Random Forest model saved: {MODEL_FILE}")
