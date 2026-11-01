
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(RAW / "orders.csv")

# Summarize each order ID without deleting or changing any rows.
summary = (
    orders.groupby("order_id", dropna=False)
    .agg(
        row_count=("order_id", "size"),
        customer_count=("customer_id", "nunique"),
        product_count=("product_id", "nunique"),
        quantity_count=("quantity", "nunique"),
        amount_count=("order_amount", "nunique"),
        timestamp_count=("order_ts", "nunique"),
        status_count=("status", "nunique"),
        statuses=("status", lambda s: " | ".join(
            sorted(s.dropna().astype(str).unique())
        )),
    )
    .reset_index()
)

repeated = summary[summary["row_count"] > 1].copy()
conflicts = repeated[repeated["status_count"] > 1].copy()

# Save full audit reports.
repeated.to_csv(REPORTS / "repeated_order_audit.csv", index=False)
conflicts.to_csv(REPORTS / "order_status_conflicts.csv", index=False)

print("=== Repeated Order Audit ===")
print(f"Total raw rows: {len(orders)}")
print(f"Unique order IDs: {orders['order_id'].nunique()}")
print(f"Order IDs with multiple rows: {len(repeated)}")
print(f"Order IDs with conflicting statuses: {len(conflicts)}")

print("\n=== Repeated IDs with Different Statuses ===")
columns = [
    "order_id", "row_count", "customer_count", "product_count",
    "quantity_count", "amount_count", "timestamp_count", "statuses"
]
print(conflicts[columns].head(20).to_string(index=False))

print("\n=== Audit Files ===")
print(REPORTS / "repeated_order_audit.csv")
print(REPORTS / "order_status_conflicts.csv")
