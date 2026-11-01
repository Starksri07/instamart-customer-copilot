
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(CLEAN / "orders_order_level.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders["status"] = orders["status"].astype(str).str.lower().str.strip()
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")

# Define the historical feature and outcome windows.
cutoff = pd.Timestamp("2025-07-01")
outcome_end = pd.Timestamp("2025-08-30")  # exclusive: includes Aug 29

# Only customers with a completed order before the cutoff are eligible.
history = orders[
    (orders["order_ts"] < cutoff)
    & (orders["status"] == "completed")
].copy()

future = orders[
    (orders["order_ts"] >= cutoff)
    & (orders["order_ts"] < outcome_end)
    & (orders["status"] == "completed")
].copy()

# Build features exclusively from the historical window.
reference_date = cutoff
features = history.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()

features["recency_days"] = (
    reference_date - features["last_order_date"]
).dt.days

# Target: 1 means no completed order during the following 60 days.
future_buyers = set(future["customer_id"].dropna().astype(str))
features["customer_id"] = features["customer_id"].astype(str)
features["lapsed_60d"] = (
    ~features["customer_id"].isin(future_buyers)
).astype(int)

features = features.drop(columns=["last_order_date"])
features.to_csv(CLEAN / "lapse_training_data.csv", index=False)

print("=== Historical Lapse Dataset ===")
print(f"Feature cutoff: {cutoff.date()}")
print(f"Outcome window: {cutoff.date()} to {outcome_end.date()} (end exclusive)")
print(f"Eligible customers: {len(features)}")
print("\nTarget counts:")
print(features["lapsed_60d"].value_counts().rename(
    index={0: "Returned within 60 days", 1: "No completed order in 60 days"}
).to_string())
print("\nTarget percentages:")
print((features["lapsed_60d"].value_counts(normalize=True) * 100).round(2).to_string())

print(f"\nSaved: {CLEAN / 'lapse_training_data.csv'}")
