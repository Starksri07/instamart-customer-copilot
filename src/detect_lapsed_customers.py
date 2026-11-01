
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

rfm = pd.read_csv(CLEAN / "customer_rfm.csv")
segments = pd.read_csv(CLEAN / "customer_segments.csv")

REFERENCE_DATE = pd.Timestamp("2025-10-01")
LAPSE_DAYS = 60

# Merge the segment label onto the RFM features.
df = rfm.merge(
    segments[["customer_id", "segment"]],
    on="customer_id",
    how="left",
    suffixes=("", "_segment")
)

# Handle a possible duplicate segment column after merging.
if "segment_segment" in df.columns:
    df["segment"] = df["segment_segment"]
    df = df.drop(columns=["segment_segment"])

df["recency_days"] = pd.to_numeric(
    df["recency_days"], errors="coerce"
)
df["frequency"] = pd.to_numeric(
    df["frequency"], errors="coerce"
)
df["monetary"] = pd.to_numeric(
    df["monetary"], errors="coerce"
)

# Lapsed customers must have bought before, but not recently.
df["lapse_status"] = "Active"
df.loc[df["frequency"].fillna(0).eq(0), "lapse_status"] = (
    "No Completed Orders"
)
df.loc[
    df["frequency"].fillna(0).gt(0)
    & df["recency_days"].ge(LAPSE_DAYS),
    "lapse_status"
] = "Potentially Lapsed"

lapsed = df[df["lapse_status"] == "Potentially Lapsed"].copy()
lapsed = lapsed.sort_values(
    ["recency_days", "monetary"],
    ascending=[False, False]
)

output = REPORTS / "lapsed_customer_list.csv"
lapsed.to_csv(output, index=False)

print("=== Lapse Analysis ===")
print(f"Reference date: {REFERENCE_DATE.date()}")
print(f"Lapse threshold: {LAPSE_DAYS} days")
print(df["lapse_status"].value_counts().to_string())
print(f"\nPotentially lapsed customers: {len(lapsed)}")
print(f"Saved: {output}")

print("\n=== Top 10 Potentially Lapsed Customers ===")
columns = [
    "customer_id", "city", "recency_days",
    "frequency", "monetary", "segment"
]
print(lapsed[columns].head(10).to_string(index=False))
