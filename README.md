# Instamart Customer Intelligence & Support Copilot

## Overview

An offline customer intelligence and support copilot built using Python, SQLite, machine learning, TF-IDF retrieval, and a simulated LLM.

The project uses synthetic Instamart customer, product, order, and support-ticket data. It combines data engineering, customer segmentation, lapse prediction, ticket triage, policy retrieval, and refund-support agents.

All LLM operations use the supplied offline `mock_llm.py`. No external AI API, paid service, or real payment system is used.

## Reproduce

### Requirements

- Python 3.11 or later
- SQLite 3.25 or later
- pandas
- numpy
- scikit-learn
- joblib

### Setup

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/Starksri07/instamart-customer-copilot.git
cd instamart-customer-copilot
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Generate and prepare data

Run these commands in order from the project root:

```powershell
python generate_data.py
python src/clean_data_v2.py
python src/resolve_orders.py
```

### Part B – Customer intelligence

```powershell
python src/build_rfm.py
python src/segment_customers.py
python src/build_lapse_training_data.py
python src/train_lapse_model_v2.py
```

### Part C – Ticket triage

```powershell
python src/ticket_triage.py
```

### Part D – Retrieval and refund copilot

```powershell
python src/rag_evaluation.py
python src/policy_assistant.py
```

Run the refund agent using the interactive prompts:

```powershell
python src/refund_agent.py
```

Run the agent comparison after completing the four ReAct scenarios configured in `data/raw/agent_scenarios.csv`:

```powershell
python src/compare_agents.py
```

### Additional tests

Set the Python import path from the project root:

```powershell
$env:PYTHONPATH = (Get-Location).Path + ";" + (Join-Path (Get-Location).Path "src")
```

Run the tests:

```powershell
python tests/test_missing_order.py
python tests/test_checkpoint.py
python tests/test_loop_guard.py
python tests/test_loop_guard_src.py
```

## Part A

### Data engineering and SQL

- Generated synthetic customer, product, order, and ticket datasets.
- Cleaned and validated input records.
- Identified invalid order amounts and duplicate records.
- Preserved repeated order IDs for further investigation.
- Resolved order-level records and flagged ambiguous order IDs.
- Stored cleaned data in SQLite.
- Prepared seven SQL analysis queries in `sql/queries.sql`.

### Dataset summary

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Raw orders | 13,084 |
| Cleaned orders | 12,847 |
| Support tickets | 164 |
| Unique order IDs | 12,309 |
| Resolved orders | 12,197 |
| Flagged order IDs | 112 |

## Part B

### Customer intelligence

- Calculated Recency, Frequency, and Monetary (RFM) features.
- Applied K-Means clustering for customer segmentation.
- Used PCA for visualization.
- Built lapse-prediction models using Logistic Regression and Random Forest.
- Applied oversampling to address class imbalance.
- Evaluated models using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

### Customer segmentation

| Segment | Customers |
|---|---:|
| High Value | 414 |
| Regular | 274 |
| Needs Attention | 269 |
| Low Engagement | 216 |
| No Completed Orders | 27 |
| **Total** | **1,200** |

### Lapse-prediction evaluation

Logistic Regression was selected based on cross-validation F1-score.

| Metric | Test result |
|---|---:|
| Accuracy | 0.7154 |
| Precision | 0.3694 |
| Recall | 0.8723 |
| F1-score | 0.5190 |
| ROC-AUC | 0.8009 |

## Part C

### Support ticket triage

- Classified tickets into billing, delivery, fraud, and technical categories.
- Masked sensitive information before processing.
- Implemented validation, retry handling, and safe fallback responses.
- Used the supplied offline mock LLM.
- Evaluated classification performance.

### Ticket classification results

All 164 tickets were processed.

| Category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Billing | 0.750 | 0.769 | 0.759 |
| Delivery | 1.000 | 0.907 | 0.951 |
| Fraud | 0.642 | 0.872 | 0.739 |
| Technical | 1.000 | 0.718 | 0.836 |

## Part D

### Policy retrieval and refund copilot

- Implemented TF-IDF policy retrieval using cosine similarity.
- Evaluated retrieval quality, including zero-overlap queries.
- Implemented order lookup and policy search tools.
- Added refund calculation and simulated refund handling.
- Implemented human escalation.
- Developed ReAct and Plan-and-Execute refund agents.
- Added checkpoint and resume functionality.
- Implemented maximum-step protection to prevent infinite agent loops.

### RAG evaluation

- Six retrieval evaluation cases passed.
- Expected policy sources were retrieved successfully.
- Zero-overlap queries returned no irrelevant results.

### Agent comparison

Four refund scenarios were evaluated.

| Scenario | Plan-and-Execute calls | ReAct calls |
|---|---:|---:|
| Cancelled order | 3 | 2 |
| Delivery within 45 minutes | 3 | 3 |
| High-value refund | 4 | 4 |
| Invalid order | 1 | 1 |
| **Total** | **11** | **10** |

Both agents passed all four expected-outcome scenarios.

### Technology stack

- Python 3.11+
- SQLite
- Pandas
- NumPy
- Scikit-learn
- Joblib
- TF-IDF
- Offline mock LLM

## Limitations and Next Steps

### Limitations

- All datasets are synthetic and do not represent real customers.
- The LLM is simulated using the supplied offline implementation.
- Refund actions are simulated; no actual money is transferred.
- The order database does not contain a delivery date, so the seven-day refund-request condition cannot be fully validated.
- Model performance is based on synthetic data and is not a production-performance guarantee.
- Agent comparison covers four selected scenarios and is not a comprehensive benchmark.

### Next steps

- Expand agent testing with more order and refund scenarios.
- Improve customer lapse prediction with additional features.
- Add more comprehensive data validation and automated tests.
- Improve human-escalation and ticket-priority routing.
- Extend refund validation when delivery-date information is available.

## Repository Structure

```text
instamart-customer-copilot/
├── data/
│   ├── raw/
│   ├── clean/
│   └── policies/
├── outputs/
├── reports/
├── sql/
│   └── queries.sql
├── src/
├── tests/
├── generate_data.py
├── mock_llm.py
├── requirements.txt
└── README.md
```

## Author

Sri Suriya R