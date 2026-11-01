
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

customers = pd.read_csv(RAW / "customers.csv")
orders = pd.read_csv(RAW / "orders.csv")

# 1. Inspect invalid order amounts
orders["amount_numeric"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

bad_amounts = orders[
    orders["amount_numeric"].isna()
    | (orders["amount_numeric"] < 0)
]

print("\n=== Invalid Order Amounts ===")
print(f"Count: {len(bad_amounts)}")
print(bad_amounts[["order_id", "order_amount"]].head(20).to_string(index=False))

# 2. Inspect invalid customer ages
customers["age_numeric"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

bad_ages = customers[
    customers["age_numeric"].isna()
    | (customers["age_numeric"] < 1)
    | (customers["age_numeric"] > 120)
]

print("\n=== Invalid Customer Ages ===")
print(f"Count: {len(bad_ages)}")
print(bad_ages[["customer_id", "age"]].head(20).to_string(index=False))

# 3. Inspect repeated order IDs
duplicate_orders = orders[
    orders.duplicated(subset=["order_id"], keep=False)
].sort_values("order_id")

print("\n=== Repeated Order IDs ===")
print(f"Rows involved: {len(duplicate_orders)}")
print(
    duplicate_orders[
        ["order_id", "customer_id", "product_id", "quantity", "status"]
    ].head(20).to_string(index=False)
)

# 4. Check whether repeated IDs contain different products
products_per_order = (
    orders.groupby("order_id")["product_id"].nunique()
)

print("\n=== Order ID Summary ===")
print(f"Unique order IDs: {orders['order_id'].nunique()}")
print(f"IDs with multiple products: {(products_per_order > 1).sum()}")
print(f"IDs with one product: {(products_per_order == 1).sum()}")

print("\nInspection complete. No source data was modified.")
