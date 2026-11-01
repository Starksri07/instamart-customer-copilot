
from pathlib import Path
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"
DB = ROOT / "data" / "instamart_v2.db"

CLEAN.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def load_csv(name):
    df = pd.read_csv(RAW / f"{name}.csv")
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("", pd.NA)

    return df


def save_csv(df, filename):
    df.to_csv(CLEAN / filename, index=False)


def main():
    customers = load_csv("customers")
    products = load_csv("products")
    orders = load_csv("orders")
    tickets = load_csv("tickets")

    audit = []

    # Customers: validate age without inventing values.
    customers["age"] = pd.to_numeric(
        customers["age"], errors="coerce"
    )
    invalid_age = customers[
        customers["age"].isna()
        | (customers["age"] < 1)
        | (customers["age"] > 120)
    ].copy()

    invalid_age.to_csv(
        REPORTS / "invalid_customer_ages.csv", index=False
    )
    customers.loc[
        (customers["age"] < 1) | (customers["age"] > 120),
        "age"
    ] = pd.NA

    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Products: validate price.
    products["price_inr"] = pd.to_numeric(
        products["price_inr"], errors="coerce"
    )
    invalid_price = products[
        products["price_inr"].isna()
        | (products["price_inr"] <= 0)
    ].copy()
    invalid_price.to_csv(
        REPORTS / "invalid_products.csv", index=False
    )

    # Orders: preserve a full audit of repeated IDs.
    repeated_ids = orders[
        orders.duplicated("order_id", keep=False)
    ].copy()
    repeated_ids.to_csv(
        REPORTS / "repeated_order_ids.csv", index=False
    )

    orders["quantity"] = pd.to_numeric(
        orders["quantity"], errors="coerce"
    )
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="coerce"
    )
    orders["order_amount"] = pd.to_numeric(
        orders["order_amount"], errors="coerce"
    )
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )

    invalid_order_amounts = orders[
        orders["order_amount"].isna()
        | (orders["order_amount"] < 0)
    ].copy()
    invalid_order_amounts.to_csv(
        REPORTS / "invalid_order_amounts.csv", index=False
    )

    # Keep invalid amounts out of the analytical order table.
    # The rejected records remain available in the audit CSV.
    valid_orders = orders[
        orders["order_amount"].notna()
        & (orders["order_amount"] >= 0)
        & orders["quantity"].notna()
        & (orders["quantity"] > 0)
        & orders["order_ts"].notna()
    ].copy()

    # Remove exact duplicate records only.
    before = len(valid_orders)
    valid_orders = valid_orders.drop_duplicates()
    exact_duplicates_removed = before - len(valid_orders)

    # Standardize common text fields.
    for col in ["city", "gender"]:
        if col in customers.columns:
            customers[col] = customers[col].str.upper()

    for col in ["category", "product_name"]:
        if col in products.columns:
            products[col] = products[col].str.strip()

    if "status" in valid_orders.columns:
        valid_orders["status"] = valid_orders["status"].str.lower()

    # Tickets: remove exact duplicates only; do not alter ticket text.
    tickets = tickets.drop_duplicates()

    # Save cleaned files.
    save_csv(customers, "customers_clean_v2.csv")
    save_csv(products, "products_clean_v2.csv")
    save_csv(valid_orders, "orders_clean_v2.csv")
    save_csv(tickets, "tickets_clean_v2.csv")

    # Save a compact cleaning summary.
    summary = pd.DataFrame([
        {
            "dataset": "customers",
            "input_rows": len(load_csv("customers")),
            "output_rows": len(customers),
            "notes": f"{len(invalid_age)} rows had missing/invalid age; invalid values set to missing",
        },
        {
            "dataset": "products",
            "input_rows": len(load_csv("products")),
            "output_rows": len(products),
            "notes": f"{len(invalid_price)} rows had invalid prices; inspect audit file",
        },
        {
            "dataset": "orders",
            "input_rows": len(load_csv("orders")),
            "output_rows": len(valid_orders),
            "notes": (
                f"{len(invalid_order_amounts)} invalid amounts; "
                f"{exact_duplicates_removed} exact duplicates removed; "
                "repeated IDs preserved for investigation"
            ),
        },
        {
            "dataset": "tickets",
            "input_rows": len(load_csv("tickets")),
            "output_rows": len(tickets),
            "notes": "Exact duplicate rows removed",
        },
    ])
    summary.to_csv(REPORTS / "cleaning_summary_v2.csv", index=False)

    # Build a separate v2 database so the earlier database is untouched.
    with sqlite3.connect(DB) as conn:
        customers.to_sql("customers", conn, if_exists="replace", index=False)
        products.to_sql("products", conn, if_exists="replace", index=False)
        valid_orders.to_sql("orders", conn, if_exists="replace", index=False)
        tickets.to_sql("tickets", conn, if_exists="replace", index=False)

    print("\n=== Cleaning Summary ===")
    print(summary.to_string(index=False))
    print(f"\nDatabase created: {DB}")
    print(f"Audit reports saved in: {REPORTS}")


if __name__ == "__main__":
    main()
