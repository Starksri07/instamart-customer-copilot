
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(CLEAN / "customers_clean_v2.csv")
orders = pd.read_csv(CLEAN / "orders_order_level.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Use completed orders only for purchase-based RFM.
completed = orders[
    orders["status"].str.lower().eq("completed")
    & orders["order_ts"].notna()
    & orders["order_amount"].notna()
    & (orders["order_amount"] >= 0)
].copy()

if completed.empty:
    raise ValueError("No valid completed orders found.")

# Fixed reference date makes the analysis reproducible.
# One day after the latest completed order.
reference_date = completed["order_ts"].max().normalize() + pd.Timedelta(days=1)

rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency_days"] = (
    reference_date - rfm["last_order_date"].dt.normalize()
).dt.days

# Include customers with no completed orders.
rfm = customers[["customer_id", "city"]].merge(
    rfm, on="customer_id", how="left"
)

rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
rfm["monetary"] = rfm["monetary"].fillna(0.0)

# Customers without a completed order have undefined recency.
rfm["recency_days"] = rfm["recency_days"].astype("Int64")

output = CLEAN / "customer_rfm.csv"
rfm.to_csv(output, index=False)

print("=== RFM Dataset Summary ===")
print(f"Reference date: {reference_date.date()}")
print(f"Customers: {len(rfm)}")
print(f"Customers with completed orders: {(rfm['frequency'] > 0).sum()}")
print(f"Customers without completed orders: {(rfm['frequency'] == 0).sum()}")
print(f"Saved: {output}")

print("\n=== RFM Sample ===")
print(rfm.head(10).to_string(index=False))
