
from pathlib import Path
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
DB = ROOT / "data" / "instamart.db"

REPORTS.mkdir(parents=True, exist_ok=True)


def load(name):
    return pd.read_csv(RAW / f"{name}.csv")


def add_check(rows, dataset, check, failed, details=""):
    rows.append({
        "dataset": dataset,
        "check": check,
        "failed_rows": int(failed),
        "details": details,
    })


def main():
    customers = load("customers")
    products = load("products")
    orders = load("orders")
    tickets = load("tickets")

    results = []

    # 1. Primary-key checks
    for name, df, key in [
        ("customers", customers, "customer_id"),
        ("products", products, "product_id"),
        ("orders", orders, "order_id"),
        ("tickets", tickets, "ticket_id"),
    ]:
        add_check(results, name, f"Missing {key}",
                  df[key].isna().sum())
        add_check(results, name, f"Duplicate {key}",
                  df[key].duplicated(keep=False).sum())

    # 2. Customer checks
    customer_age = pd.to_numeric(customers["age"], errors="coerce")
    add_check(results, "customers", "Missing or invalid age",
              customer_age.isna().sum())
    add_check(results, "customers", "Age outside 1-120",
              ((customer_age < 1) | (customer_age > 120)).sum())

    signup_date = pd.to_datetime(customers["signup_date"], errors="coerce")
    add_check(results, "customers", "Invalid signup date",
              signup_date.isna().sum())

    # 3. Product checks
    price = pd.to_numeric(products["price_inr"], errors="coerce")
    add_check(results, "products", "Missing or invalid price",
              price.isna().sum())
    add_check(results, "products", "Price not positive",
              (price <= 0).sum())

    # 4. Order checks
    quantity = pd.to_numeric(orders["quantity"], errors="coerce")
    amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    discount = pd.to_numeric(orders["discount_pct"], errors="coerce")
    order_date = pd.to_datetime(orders["order_ts"], errors="coerce")

    add_check(results, "orders", "Missing customer ID",
              orders["customer_id"].isna().sum())
    add_check(results, "orders", "Missing product ID",
              orders["product_id"].isna().sum())
    add_check(results, "orders", "Invalid quantity",
              (quantity.isna() | (quantity <= 0)).sum())
    add_check(results, "orders", "Invalid order amount",
              (amount.isna() | (amount < 0)).sum())
    add_check(results, "orders", "Invalid discount percentage",
              (discount.isna() | (discount < 0) | (discount > 100)).sum())
    add_check(results, "orders", "Invalid order timestamp",
              order_date.isna().sum())

    # 5. Foreign-key checks
    customer_ids = set(customers["customer_id"].dropna())
    product_ids = set(products["product_id"].dropna())
    order_ids = set(orders["order_id"].dropna())

    add_check(
        results, "orders", "Customer IDs not found in customers",
        (~orders["customer_id"].isin(customer_ids)).sum()
    )
    add_check(
        results, "orders", "Product IDs not found in products",
        (~orders["product_id"].isin(product_ids)).sum()
    )
    add_check(
        results, "tickets", "Customer IDs not found in customers",
        (~tickets["customer_id"].isin(customer_ids)).sum()
    )
    add_check(
        results, "tickets", "Order IDs not found in orders",
        (
            tickets["order_id"].notna()
            & ~tickets["order_id"].isin(order_ids)
        ).sum()
    )

    # 6. Exact duplicate rows
    for name, df in [
        ("customers", customers),
        ("products", products),
        ("orders", orders),
        ("tickets", tickets),
    ]:
        add_check(results, name, "Exact duplicate rows",
                  df.duplicated().sum())

    report = pd.DataFrame(results)
    output = REPORTS / "validation_report.csv"
    report.to_csv(output, index=False)

    print("\n=== Data Validation Report ===")
    print(report.to_string(index=False))
    print(f"\nSaved report: {output}")

    # Confirm that SQLite tables match the cleaned CSV row counts
    print("\n=== SQLite Row Counts ===")
    with sqlite3.connect(DB) as connection:
        for name in ["customers", "products", "orders", "tickets"]:
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{name}"'
            ).fetchone()[0]
            print(f"{name}: {count}")


if __name__ == "__main__":
    main()
