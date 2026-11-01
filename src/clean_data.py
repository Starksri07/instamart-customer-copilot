
from pathlib import Path
import sqlite3
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
REPORT_DIR = ROOT / "reports"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = ROOT / "data" / "instamart.db"


def load_csv(filename):
    """Load a raw CSV file and standardize column names."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def clean_dataframe(df, table_name):
    """Apply general cleaning and report basic data-quality checks."""
    df = df.copy()

    # Remove completely duplicated rows
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # Standardize text fields and convert blank strings to missing values
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace("", pd.NA)

    report = {
        "table": table_name,
        "rows_after_cleaning": len(df),
        "duplicate_rows_removed": duplicate_rows,
        "missing_cells": int(df.isna().sum().sum()),
    }
    return df, report


def main():
    datasets = {
        "customers": "customers.csv",
        "products": "products.csv",
        "orders": "orders.csv",
        "tickets": "tickets.csv",
    }

    cleaned = {}
    quality_reports = []

    for table_name, filename in datasets.items():
        print(f"Loading {filename}...")
        df = load_csv(filename)
        df, report = clean_dataframe(df, table_name)

        cleaned[table_name] = df
        quality_reports.append(report)

        # Save cleaned CSV
        output_path = CLEAN_DIR / f"{table_name}_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df)} rows -> {output_path.relative_to(ROOT)}")

    # Write cleaned datasets to SQLite
    with sqlite3.connect(DB_PATH) as connection:
        for table_name, df in cleaned.items():
            df.to_sql(table_name, connection, if_exists="replace", index=False)

        # Capture actual database row counts
        database_counts = []
        for table_name in cleaned:
            count = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
            database_counts.append({
                "table": table_name,
                "sqlite_rows": count,
            })

    # Save data-quality and database validation reports
    quality_df = pd.DataFrame(quality_reports)
    quality_df.to_csv(REPORT_DIR / "data_quality_report.csv", index=False)

    counts_df = pd.DataFrame(database_counts)
    counts_df.to_csv(REPORT_DIR / "sqlite_validation_report.csv", index=False)

    print("\nData cleaning complete.")
    print("\nData quality report:")
    print(quality_df.to_string(index=False))

    print("\nSQLite validation:")
    print(counts_df.to_string(index=False))
    print(f"\nDatabase created: {DB_PATH}")


if __name__ == "__main__":
    main()
