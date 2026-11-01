
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

rfm = pd.read_csv(CLEAN / "customer_rfm.csv")

features = ["recency_days", "frequency", "monetary"]

# Customers with no completed orders cannot be clustered using RFM.
has_orders = rfm["frequency"] > 0
active = rfm.loc[has_orders].copy()

if len(active) < 4:
    raise ValueError("Not enough customers with completed orders.")

X = active[features].copy()

if X.isna().any().any():
    raise ValueError("RFM features contain missing values.")

# Log transforms reduce the influence of very large values.
X_log = np.log1p(X.clip(lower=0))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

active["cluster"] = model.fit_predict(X_scaled)

# Give clusters descriptive labels based on their averages.
cluster_summary = (
    active.groupby("cluster")
    .agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    )
    .round(2)
)

# Lower recency and higher frequency/spend generally indicate stronger customers.
# Use a transparent ranking score to create readable labels.
cluster_summary["priority_score"] = (
    -cluster_summary["avg_recency_days"].rank(pct=True)
    + cluster_summary["avg_frequency"].rank(pct=True)
    + cluster_summary["avg_monetary"].rank(pct=True)
)

ranked_clusters = cluster_summary["priority_score"].rank(
    ascending=False, method="first"
)

label_names = {
    1: "High Value",
    2: "Regular",
    3: "Needs Attention",
    4: "Low Engagement",
}

cluster_labels = {
    cluster: label_names[int(rank)]
    for cluster, rank in ranked_clusters.items()
}

active["segment"] = active["cluster"].map(cluster_labels)

# Add customers with no completed orders back into the output.
rfm["cluster"] = pd.Series(pd.NA, index=rfm.index, dtype="Int64")
rfm["segment"] = "No Completed Orders"

rfm.loc[active.index, "cluster"] = active["cluster"].astype("Int64")
rfm.loc[active.index, "segment"] = active["segment"]

output = CLEAN / "customer_segments.csv"
rfm.to_csv(output, index=False)

summary = (
    rfm.groupby("segment", dropna=False)
    .agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    )
    .round(2)
    .sort_values("customers", ascending=False)
)

summary.to_csv(REPORTS / "customer_segment_summary.csv")

print("=== Customer Segment Summary ===")
print(summary.to_string())
print(f"\nSaved customer segments: {output}")
print(f"Saved summary: {REPORTS / 'customer_segment_summary.csv'}")
