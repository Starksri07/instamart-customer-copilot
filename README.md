\# Instamart Customer Intelligence \& Support Copilot



An offline, AI-assisted customer intelligence and support system built with Python, SQLite, machine learning, TF-IDF retrieval, and a simulated LLM.



\## Project Overview



This project analyzes synthetic Instamart customer, order, product, and support-ticket data. It combines customer segmentation, lapse prediction, automated ticket triage, policy retrieval, and refund-support agents.



The project uses an offline mock LLM. It does not connect to real customer accounts, payment systems, or external AI APIs.



\## Project Modules



\### Part A – Data Engineering and SQL

\- Clean and validate customer, product, order, and ticket datasets.

\- Store processed data in SQLite.

\- Execute seven SQL analysis queries.

\- Generate cleaned datasets for downstream analysis.



\### Part B – Customer Intelligence

\- Calculate Recency, Frequency, and Monetary (RFM) features.

\- Perform customer segmentation using K-Means.

\- Use PCA for visualization.

\- Train and compare Logistic Regression and Random Forest lapse-prediction models.

\- Evaluate using precision, recall, F1-score, ROC-AUC, and confusion matrix.

\- Apply oversampling to address class imbalance.



\### Part C – Support Ticket Triage

\- Classify support tickets into delivery, billing, fraud, and technical categories.

\- Mask sensitive information before processing.

\- Apply retries, validation, and safe fallback handling.

\- Evaluate classification performance.



\### Part D – Policy Retrieval and Refund Copilot

\- Retrieve relevant policy text using TF-IDF and cosine similarity.

\- Evaluate retrieval quality, including zero-overlap queries.

\- Implement order lookup, policy search, refund calculation, simulated refund, and human escalation tools.

\- Implement ReAct and Plan-and-Execute refund agents.

\- Support checkpoint and resume testing.

\- Compare agent outcomes and tool-call counts.



\## Technology Stack



\- Python 3.11+

\- SQLite

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib

\- TF-IDF

\- Offline mock LLM



\## Dataset



The project uses synthetic data.



| Dataset | Records |

|---|---:|

| Customers | 1,200 |

| Products | 40 |

| Raw orders | 13,084 |

| Cleaned orders | 12,847 |

| Support tickets | 164 |



\## Model Evaluation



The selected lapse-prediction model was Logistic Regression.



| Metric | Test result |

|---|---:|

| Accuracy | 0.7154 |

| Precision | 0.3694 |

| Recall | 0.8723 |

| F1-score | 0.5190 |

| ROC-AUC | 0.8009 |



The model was selected using cross-validation F1-score.



\## Ticket Triage Evaluation



All 164 tickets were processed successfully.



| Category | Precision | Recall | F1-score |

|---|---:|---:|---:|

| Billing | 0.750 | 0.769 | 0.759 |

| Delivery | 1.000 | 0.907 | 0.951 |

| Fraud | 0.642 | 0.872 | 0.739 |

| Technical | 1.000 | 0.718 | 0.836 |



\## Agent Comparison



Four refund scenarios were tested with both agents.



| Scenario | Plan-and-Execute calls | ReAct calls |

|---|---:|---:|

| Cancelled order | 3 | 2 |

| Delivery within 45 minutes | 3 | 3 |

| High-value refund | 4 | 4 |

| Invalid order | 1 | 1 |

| \*\*Total\*\* | \*\*11\*\* | \*\*10\*\* |



Both agents passed all four expected-outcome tests.



\## Running the Project



Create and activate a virtual environment:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

python -m pip install pandas numpy scikit-learn joblib

```



Run the modules from the project root:



```powershell

python src/sql\_analysis.py

python src/ticket\_triage.py

python src/rag\_evaluation.py

python src/policy\_assistant.py

python src/refund\_agent.py

python src/compare\_agents.py

```



Use the generated datasets and SQLite databases supplied in the project. Do not regenerate the data unless you intend to rebuild the database and dependent outputs.



\## Important Limitations



\- All datasets are synthetic.

\- The LLM is simulated using the supplied offline mock implementation.

\- Refund actions are recorded as simulated actions; no real money is transferred.

\- The order database does not contain a delivery date, so the seven-day refund-request condition cannot be fully validated.

\- Model results are based on a synthetic dataset and should not be interpreted as production performance.

\- Agent comparison covers four selected scenarios and is not a comprehensive benchmark.



\## Repository Structure



```text

instamart-customer-copilot/

├── data/

│   ├── raw/

│   ├── clean/

│   └── policies/

├── outputs/

├── reports/

├── sql/

├── src/

├── state/

├── generate\_data.py

├── mock\_llm.py

└── README.md

```



\## Author



Sri Suriya R



