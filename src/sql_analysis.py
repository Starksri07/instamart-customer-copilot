
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "instamart_v2.db"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

queries = {
    "orders_by_status": """
        SELECT LOWER(status) AS status,
               COUNT(*) AS order_rows
        FROM orders
        GROUP BY LOWER(status)
        ORDER BY order_rows DESC;
    """,

    "top_products_by_quantity": """
        SELECT p.product_name,
               SUM(o.quantity) AS units_sold,
               COUNT(*) AS order_rows
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        WHERE LOWER(o.status) = 'completed'
        GROUP BY p.product_id, p.product_name
        ORDER BY units_sold DESC
        LIMIT 10;
    """,

    "city_customer_counts": """
        SELECT UPPER(c.city) AS city,
               COUNT(*) AS customers
        FROM customers c
        GROUP BY UPPER(c.city)
        ORDER BY customers DESC;
    """,

    "completed_sales_by_city": """
        SELECT UPPER(c.city) AS city,
               ROUND(SUM(o.order_amount), 2) AS sales_value,
               COUNT(*) AS completed_order_rows
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE LOWER(o.status) = 'completed'
        GROUP BY UPPER(c.city)
        ORDER BY sales_value DESC;
    """,

    "customer_purchase_summary": """
        SELECT c.customer_id,
               UPPER(c.city) AS city,
               COUNT(*) AS completed_order_rows,
               ROUND(SUM(o.order_amount), 2) AS total_spend
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE LOWER(o.status) = 'completed'
        GROUP BY c.customer_id, UPPER(c.city)
        ORDER BY total_spend DESC
        LIMIT 20;
    """
}

with sqlite3.connect(DB) as conn:
    for name, query in queries.items():
        df = pd.read_sql_query(query, conn)
        output = REPORTS / f"{name}.csv"
        df.to_csv(output, index=False)

        print(f"\n=== {name.replace('_', ' ').title()} ===")
        print(df.head(10).to_string(index=False))
        print(f"Saved: {output}")

print("\nSQL analysis completed.")
