
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "instamart_v2.db"

with sqlite3.connect(DB) as conn:
    print("=== Clean Database Verification ===")

    for table in ["customers", "products", "orders", "tickets"]:
        count = pd.read_sql_query(
            f'SELECT COUNT(*) AS total FROM "{table}"',
            conn
        )["total"].iloc[0]

        print(f"{table}: {count} rows")

    print("\n=== Order Status Distribution ===")
    status = pd.read_sql_query(
        """
        SELECT LOWER(status) AS status, COUNT(*) AS total
        FROM orders
        GROUP BY LOWER(status)
        ORDER BY total DESC
        """,
        conn
    )
    print(status.to_string(index=False))

    print("\n=== Order Amount Check ===")
    amount_check = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS total_orders,
            SUM(CASE
                WHEN order_amount IS NULL THEN 1
                WHEN order_amount < 0 THEN 1
                ELSE 0
            END) AS invalid_amounts
        FROM orders
        """,
        conn
    )
    print(amount_check.to_string(index=False))

print("\nVerification complete.")
