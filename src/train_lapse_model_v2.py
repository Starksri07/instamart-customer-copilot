
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from model_utils import RandomOversampledClassifier


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "clean" / "lapse_training_data.csv"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "src"

OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

FEATURES = [
    "recency_days",
    "frequency",
    "monetary"
]

TARGET = "lapsed_60d"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 4. BASELINE MODEL
# --------------------------------------------------

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_metrics = {
    "model": "Majority Baseline",
    "accuracy": accuracy_score(y_test, baseline_predictions),
    "precision": precision_score(
        y_test, baseline_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, baseline_predictions, zero_division=0
    ),
    "f1_score": f1_score(
        y_test, baseline_predictions, zero_division=0
    ),
    "roc_auc": 0.5
}


# --------------------------------------------------
# 5. DEFINE MODELS WITH OVERSAMPLING
# --------------------------------------------------

lr_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight=None
    ))
])

lr_model = RandomOversampledClassifier(
    estimator=lr_pipeline,
    random_state=RANDOM_STATE
)

rf_model = RandomOversampledClassifier(
    estimator=rf_pipeline,
    random_state=RANDOM_STATE
)


# --------------------------------------------------
# 6. GRID SEARCH
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

lr_parameters = {
    "estimator__classifier__C": [0.01, 0.1, 1, 10],
    "estimator__classifier__solver": ["liblinear", "lbfgs"]
}

rf_parameters = {
    "estimator__classifier__n_estimators": [100, 200],
    "estimator__classifier__max_depth": [3, 5, 8, None],
    "estimator__classifier__min_samples_leaf": [2, 5, 10]
}

lr_search = GridSearchCV(
    estimator=lr_model,
    param_grid=lr_parameters,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True
)

rf_search = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_parameters,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True
)

print("\nTraining Logistic Regression...")
lr_search.fit(X_train, y_train)

print("\nTraining Random Forest...")
rf_search.fit(X_train, y_train)

print("\nLogistic Regression best parameters:")
print(lr_search.best_params_)
print("Logistic Regression CV F1:", lr_search.best_score_)

print("\nRandom Forest best parameters:")
print(rf_search.best_params_)
print("Random Forest CV F1:", rf_search.best_score_)


# --------------------------------------------------
# 7. SELECT MODEL USING CV SCORE ONLY
# --------------------------------------------------

if lr_search.best_score_ >= rf_search.best_score_:
    best_name = "Logistic Regression"
    best_model = lr_search.best_estimator_
else:
    best_name = "Random Forest"
    best_model = rf_search.best_estimator_

print("\nSelected model based on CV F1:", best_name)


# --------------------------------------------------
# 8. EVALUATE SELECTED MODEL ON TEST DATA
# --------------------------------------------------

test_predictions = best_model.predict(X_test)
test_probabilities = best_model.predict_proba(X_test)[:, 1]

selected_metrics = {
    "model": best_name,
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1_score": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, test_probabilities)
}

print("\nSelected model test metrics:")
for metric, value in selected_metrics.items():
    if metric != "model":
        print(f"{metric}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


# --------------------------------------------------
# 9. SAVE MODEL COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame([
    baseline_metrics,
    {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(
            y_test, lr_search.predict(X_test)
        ),
        "precision": precision_score(
            y_test, lr_search.predict(X_test), zero_division=0
        ),
        "recall": recall_score(
            y_test, lr_search.predict(X_test), zero_division=0
        ),
        "f1_score": f1_score(
            y_test, lr_search.predict(X_test), zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, lr_search.predict_proba(X_test)[:, 1]
        )
    },
    {
        "model": "Random Forest",
        "accuracy": accuracy_score(
            y_test, rf_search.predict(X_test)
        ),
        "precision": precision_score(
            y_test, rf_search.predict(X_test), zero_division=0
        ),
        "recall": recall_score(
            y_test, rf_search.predict(X_test), zero_division=0
        ),
        "f1_score": f1_score(
            y_test, rf_search.predict(X_test), zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, rf_search.predict_proba(X_test)[:, 1]
        )
    }
])

comparison.to_csv(
    REPORT_DIR / "lapse_model_comparison_v2.csv",
    index=False
)


# --------------------------------------------------
# 10. FEATURE IMPORTANCE / COEFFICIENTS
# --------------------------------------------------

fitted_pipeline = best_model.estimator_

imputer = fitted_pipeline.named_steps["imputer"]
classifier = fitted_pipeline.named_steps["classifier"]

imputed_features = imputer.transform(X_train)

if best_name == "Random Forest":
    importance_values = classifier.feature_importances_

    feature_analysis = pd.DataFrame({
        "feature": FEATURES,
        "importance": importance_values
    })

    feature_analysis = feature_analysis.sort_values(
        "importance", ascending=False
    )

else:
    coefficient_values = classifier.coef_[0]

    feature_analysis = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": coefficient_values,
        "absolute_coefficient": np.abs(coefficient_values)
    })

    feature_analysis = feature_analysis.sort_values(
        "absolute_coefficient", ascending=False
    )

feature_analysis.to_csv(
    REPORT_DIR / "lapse_feature_analysis_v2.csv",
    index=False
)

print("\nFeature Analysis:")
print(feature_analysis)


# --------------------------------------------------
# 11. SAVE CLASSIFICATION REPORT
# --------------------------------------------------

report_text = classification_report(
    y_test,
    test_predictions,
    zero_division=0
)

with open(
    REPORT_DIR / "lapse_classification_report_v2.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("Selected model: " + best_name + "\n\n")
    file.write(report_text)


# --------------------------------------------------
# 12. SAVE SELECTED MODEL
# --------------------------------------------------

model_path = MODEL_DIR / "lapse_model_v2.joblib"

joblib.dump(best_model, model_path)

print("\nModel saved:", model_path)
print("Comparison saved:", REPORT_DIR / "lapse_model_comparison_v2.csv")
print("Feature analysis saved:", REPORT_DIR / "lapse_feature_analysis_v2.csv")
print("Classification report saved:", REPORT_DIR / "lapse_classification_report_v2.txt")

print("\nPART B MODEL TRAINING COMPLETED SUCCESSFULLY.")
