
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

model_path = ROOT / "src" / "lapse_model.joblib"
rfm_path = CLEAN / "customer_rfm.csv"
segments_path = CLEAN / "customer_segments.csv"

saved = joblib.load(model_path)
model = saved["model"]
features = saved["features"]

rfm = pd.read_csv(rfm_path)
segments = pd.read_csv(segments_path)

df = rfm.merge(
    segments[["customer_id", "segment"]],
    on="customer_id",
    how="left",
    suffixes=("", "_segment"),
)

if "segment_segment" in df.columns:
    df["segment"] = df["segment_segment"]
    df = df.drop(columns=["segment_segment"])

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Model scores estimate risk under the historical training setup.
df["lapse_risk"] = model.predict_proba(df[features])[:, 1]

df["risk_band"] = pd.cut(
    df["lapse_risk"],
    bins=[-0.001, 0.30, 0.60, 1.001],
    labels=["Low", "Medium", "High"],
).astype("object")

df.loc[
    df["frequency"].fillna(0).eq(0),
    "risk_band"
] = "No Completed Orders"

df = df.sort_values(
    ["lapse_risk", "recency_days"],
    ascending=[False, False],
)

output = REPORTS / "customer_lapse_risk_scores.csv"
df.to_csv(output, index=False)

print("=== Customer Lapse Risk Scoring ===")
print(f"Customers scored: {len(df)}")
print("\nRisk-band counts:")
print(df["risk_band"].value_counts(dropna=False).to_string())

print("\n=== Top 15 Customers by Model Risk ===")
cols = [
    "customer_id",
    "city",
    "segment",
    "recency_days",
    "frequency",
    "monetary",
    "lapse_risk",
    "risk_band",
]
print(df[cols].head(15).to_string(index=False))

print(f"\nSaved: {output}")
