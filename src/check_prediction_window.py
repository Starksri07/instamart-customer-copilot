
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

orders = pd.read_csv(RAW / "orders.csv")
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
orders["status"] = orders["status"].astype(str).str.lower().str.strip()

print("=== Raw Order Date Range ===")
print("Earliest:", orders["order_ts"].min())
print("Latest:  ", orders["order_ts"].max())

print("\n=== Rows by Month and Status ===")
orders["month"] = orders["order_ts"].dt.to_period("M").astype(str)

monthly = (
    orders.groupby(["month", "status"])
    .size()
    .unstack(fill_value=0)
)

print(monthly.to_string())

print("\n=== Distinct Customers by Month ===")
monthly_customers = (
    orders.dropna(subset=["order_ts"])
    .groupby(orders["order_ts"].dt.to_period("M"))["customer_id"]
    .nunique()
)

print(monthly_customers.to_string())
