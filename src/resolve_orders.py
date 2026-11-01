
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"

CLEAN.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(RAW / "orders.csv")
orders.columns = orders.columns.str.strip()

for col in orders.select_dtypes(include=["object", "str"]).columns:
    orders[col] = orders[col].str.strip()

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="coerce"
)
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)
orders["status"] = orders["status"].str.lower()

resolved = []
ambiguous = []

identity_columns = [
    "customer_id", "product_id", "quantity", "discount_pct"
]

for order_id, group in orders.groupby("order_id", dropna=False):
    reasons = []

    # An order must have a valid ID and timestamp.
    if pd.isna(order_id):
        reasons.append("Missing order ID")

    if group["order_ts"].isna().any():
        reasons.append("Missing or invalid timestamp")

    # All records for an order should describe the same purchase.
    for col in identity_columns:
        if group[col].isna().any() or group[col].nunique(dropna=False) != 1:
            reasons.append(f"Inconsistent {col}")

    valid_amounts = group["order_amount"].dropna().unique()
    if len(valid_amounts) > 1:
        reasons.append("Inconsistent order amounts")

    # Latest timestamp represents the latest recorded status.
    dated = group.dropna(subset=["order_ts"])
    if dated.empty:
        reasons.append("No usable timestamp")
        ambiguous.append({
            "order_id": order_id,
            "row_count": len(group),
            "reasons": "; ".join(sorted(set(reasons))),
        })
        continue

    latest_time = dated["order_ts"].max()
    latest_rows = dated[dated["order_ts"] == latest_time]

    latest_statuses = latest_rows["status"].dropna().unique()

    if len(latest_statuses) != 1:
        reasons.append("Conflicting statuses at latest timestamp")

    if reasons:
        ambiguous.append({
            "order_id": order_id,
            "row_count": len(group),
            "reasons": "; ".join(sorted(set(reasons))),
        })
        continue

    # Select the latest event only after checking for ambiguity.
    chosen = latest_rows.iloc[-1].copy()

    if pd.isna(chosen["order_amount"]) or chosen["order_amount"] < 0:
        reasons.append("Invalid amount in latest record")
        ambiguous.append({
            "order_id": order_id,
            "row_count": len(group),
            "reasons": "; ".join(reasons),
        })
        continue

    if pd.isna(chosen["quantity"]) or chosen["quantity"] <= 0:
        ambiguous.append({
            "order_id": order_id,
            "row_count": len(group),
            "reasons": "Invalid quantity",
        })
        continue

    resolved.append(chosen)

resolved_df = pd.DataFrame(resolved)
ambiguous_df = pd.DataFrame(
    ambiguous, columns=["order_id", "row_count", "reasons"]
)

resolved_df.to_csv(
    CLEAN / "orders_order_level.csv", index=False
)
ambiguous_df.to_csv(
    REPORTS / "ambiguous_orders.csv", index=False
)

print("=== Order Resolution Summary ===")
print(f"Raw rows: {len(orders)}")
print(f"Unique order IDs: {orders['order_id'].nunique()}")
print(f"Resolved orders: {len(resolved_df)}")
print(f"Flagged order IDs: {len(ambiguous_df)}")

if not resolved_df.empty:
    print("\n=== Resolved Status Distribution ===")
    print(
        resolved_df["status"]
        .value_counts(dropna=False)
        .to_string()
    )

print(f"\nResolved dataset: {CLEAN / 'orders_order_level.csv'}")
print(f"Ambiguity report: {REPORTS / 'ambiguous_orders.csv'}")
print("\nRaw data was not modified.")
