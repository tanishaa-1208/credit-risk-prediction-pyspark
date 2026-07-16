# Credit Risk Prediction using PySpark
An end-to-end Spark ML pipeline for predicting loan default risk, built and evaluated on two real-world credit datasets. The project compares classical Spark ML classifiers against gradient boosting models (XGBoost, LightGBM, CatBoost) to identify high-risk borrowers.

# Problem Statement
Lenders - banks, fintech apps, and NBFCs — need to estimate the probability that a loan applicant will default before extending credit. Misjudging this risk either costs the lender money (approving bad loans) or costs them business (rejecting good applicants). This project builds a classification pipeline to predict default risk using demographic, financial, and credit history features, and benchmarks multiple models to find the best-performing approach.

# Datasets
**1. Credit Risk Dataset (~3,000 records, 12 attributes)**
Demographic and loan-related attributes (age, income, employment length, loan amount, interest rate, loan intent, loan grade) used to predict loan default (loan_status).
**Source:** Kaggle – Credit Risk Prediction
**2. Credit Card Clients Dataset (~30,000 records, 24 attributes)**
Financial and repayment behaviour data (credit limit, bill amounts, payment amounts, payment status history) used to predict next-month default (default.payment.next.month).
**Source:** Kaggle – Credit Card Clients Data


# Approach
- **Data Preprocessing:** Missing value imputation (median for numeric, mode for categorical), categorical encoding via StringIndexer + OneHotEncoder, feature scaling with StandardScaler
- **Feature Engineering:** Engineered domain-specific features including credit utilisation ratio (balance/credit limit) and other derived financial ratios
- **Pipeline:** Built using Spark ML's Pipeline API for reproducible, scalable preprocessing
- **Models Trained:**
  **1. Spark ML:** Logistic Regression, Decision Tree, Random Forest, Gradient-Boosted Trees (GBT)
  **2. Gradient Boosting (via scikit-learn interop):** XGBoost, LightGBM, CatBoost, AdaBoost
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, with confusion matrices and ROC/PR curves for the top-performing model on each dataset


# Tech Stack
PySpark (Spark ML) · XGBoost · LightGBM · CatBoost · scikit-learn · pandas · NumPy · Matplotlib / Seaborn

# Repository Structure

├── credit_risk_prediction_pyspark.ipynb   # Main notebook: full pipeline, training, evaluation
├── data/
│   ├── credit_risk_dataset.csv
│   └── credit_card_clients_dataset.csv
├── images/                                 # Output plots (ROC curves, confusion matrices, results tables)
└── README.md

# How to Run
- Clone the repo and open the notebook in Google Colab or Jupyter

- Run the setup cell to install Spark, PySpark, and boosting libraries (xgboost, lightgbm, catboost)

- Place the two dataset CSVs in the working directory (or update the file paths in the data-loading cells)

- Run all cells sequentially — the notebook builds separate pipelines for each dataset and outputs a comparison table of all models ranked by ROC-AUC


# Results
**Dataset 1 — Credit Card Clients (Performance Metrics)**

a. LightGBM — Accuracy: 0.8155 | Precision: 0.6545 | Recall: 0.3853 | F1-Score: 0.4850 | ROC-AUC: 0.7806 (best)

b. AdaBoost — Accuracy: 0.8141 | Precision: 0.6681 | Recall: 0.3497 | F1-Score: 0.4591 | ROC-AUC: 0.7794

c. Gradient Boosted Trees (GBT) — Accuracy: 0.8182 | Precision: 0.6597 | Recall: 0.4001 | F1-Score: 0.8008 | ROC-AUC: 0.7780

d. CatBoost — Accuracy: 0.8176 | Precision: 0.6617 | Recall: 0.3920 | F1-Score: 0.4923 | ROC-AUC: 0.7739

e. Random Forest — Accuracy: 0.8198 | Precision: 0.6938 | Recall: 0.3601 | F1-Score: 0.7972 | ROC-AUC: 0.7686

f. XGBoost — Accuracy: 0.8074 | Precision: 0.6157 | Recall: 0.3890 | F1-Score: 0.4768 | ROC-AUC: 0.7682


**Best model (Dataset 1):** LightGBM, selected for highest ROC-AUC. Its confusion matrix, ROC curve, and precision-recall curve confirmed strong separation between defaulters and non-defaulters.






**Dataset 2 — Loan Approval/Default (Performance Metrics)**

a. LightGBM — Accuracy: 0.9379 | Precision: 0.9737 | Recall: 0.7269 | F1-Score: 0.8324 | ROC-AUC: 0.9475 (best)

b. CatBoost — Accuracy: 0.9371 | Precision: 0.9745 | Recall: 0.7226 | F1-Score: 0.8299 | ROC-AUC: 0.9423

c. GBT — Accuracy: 0.9302 | Recall: 0.7066 | F1-Score: 0.9262 | ROC-AUC: 0.9279

d. XGBoost — Accuracy: 0.9365 | Precision: 0.9488 | Recall: 0.7407 | F1-Score: 0.8320

e. AdaBoost — Accuracy: 0.8881 | Precision: 0.7919 | F1-Score: 0.7087 | ROC-AUC: 0.9015

f. Random Forest — Accuracy: 0.9097 | Precision: 0.9703 | Recall: 0.5926 | F1-Score: 0.9010


**Best model (Dataset 2):** LightGBM, with the strongest balance of Accuracy (0.938), Precision (0.974), and ROC-AUC (0.948) — indicating excellent ability to identify defaulters while minimising false positives.



# Key Findings
1. LightGBM was the best-performing model on both datasets, consistently achieving the highest ROC-AUC by capturing non-linear, interaction-heavy relationships between financial features (credit utilisation, repayment history, income ratios) and default risk more effectively than linear or single-tree models.
2. Boosting models (LightGBM, CatBoost, XGBoost, AdaBoost) generally outperformed simpler Spark ML classifiers, reflecting their ability to correct errors and handle mixed categorical-numerical data iteratively.
3. Feature engineering (credit utilisation ratio, DTI/PTI-style derived features) combined with proper categorical encoding (StringIndexer + OneHotEncoder) and scaling (StandardScaler) was essential to getting consistent performance across both differently structured datasets.


# Author
Tanisha Gupta — B.Tech Computer Science (Data Science), Bennett University
