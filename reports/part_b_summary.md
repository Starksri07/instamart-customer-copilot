# Part B — Customer Segmentation and Lapse Prediction

## Objective
Identify customer behavior patterns and develop a baseline model to help prioritize customer retention activities.

## Customer Segmentation
- Total customers analyzed: 1,200
- High Value: 414
- Regular: 274
- Needs Attention: 269
- Low Engagement: 216
- No Completed Orders: 27

Segmentation was performed using K-Means clustering on transformed RFM (Recency, Frequency, Monetary) features.

## Rule-Based Lapse Detection
A customer with at least one completed order was classified as potentially lapsed when the customer had not placed a completed order for 60 or more days.

- Active: 934
- Potentially Lapsed: 239
- No Completed Orders: 27

Reference date: October 1, 2025.

## Baseline Machine-Learning Experiment
A Random Forest classifier was compared with a majority-class baseline using a stratified random holdout.

- Eligible historical customers: 1,066
- Returned within 60 days: 880
- No completed order within 60 days: 186
- Random Forest test accuracy: 74.5%
- Random Forest ROC-AUC: 0.785
- Recall for potentially lapsed customers: 83.0%
- Precision for potentially lapsed customers: 39.4%

The majority-class baseline achieved 82.4% accuracy but failed to identify any potentially lapsed customers.

## Generated Outputs
- `data/clean/customer_rfm.csv`
- `data/clean/customer_segments.csv`
- `data/clean/lapse_training_data.csv`
- `reports/customer_segment_summary.csv`
- `reports/lapsed_customer_list.csv`
- `reports/lapse_model_comparison.csv`
- `reports/customer_lapse_risk_scores.csv`
- `src/lapse_model.joblib`

## Limitations
The data is synthetic and covers April through September 2025. The reported model metrics come from a random holdout within one historical dataset and require temporal validation before real-world use. The customer risk-scoring output also needs a consistent scoring cutoff before it can be treated as a valid future forecast.