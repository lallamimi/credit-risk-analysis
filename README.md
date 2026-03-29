# Credit Risk Analysis & Prediction

<p align="center">
  <b>Machine Learning project for credit default prediction with real financial impact</b><br>
  <i>From data preprocessing to business-oriented risk estimation</i>
</p>

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- React
- FastAPI

---

## Project Overview

This project focuses on predicting the probability of loan default using machine learning techniques.

Beyond standard classification, it introduces a business perspective through financial risk estimation using:

> **Expected Loss (EL) = PD × LGD × EAD**

Where:
- **PD**: Probability of Default
- **LGD**: Loss Given Default
- **EAD**: Exposure at Default

This makes the project more aligned with real banking and fintech decision systems, where predictions are used to support lending decisions and risk management.

---

## Objectives

- Predict loan default risk
- Handle imbalanced data
- Build a complete machine learning pipeline
- Compare multiple classification models
- Translate predictions into business insights
- Provide an interactive frontend dashboard

---

## Dataset

- Credit Risk Dataset
- Around 32,000 observations
- Target variable: `loan_status`

### Main Features

- Borrower age and income
- Employment length
- Loan amount
- Interest rate
- Home ownership
- Credit history length
- Loan intent / loan grade

---

## Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Categorical encoding
- Feature scaling
- Train / test split

### Feature Engineering

- Financial ratios such as loan-to-income
- Derived variables for stronger predictive power

### Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Bagging
- XGBoost
- LightGBM

---

## Model Evaluation

Because the dataset is imbalanced, the main evaluation metric is:

- **F1-score**

Additional metrics include:
- Accuracy
- Precision
- Recall
- Confusion Matrix

Special attention is given to **recall**, since missing risky borrowers may lead to financial loss.

---

## Business Impact

This project connects machine learning with real-world financial decision-making.

It can help:
- Identify high-risk borrowers
- Estimate potential financial losses
- Support credit approval decisions
- Improve transparency through model explanations

The objective is not only prediction, but also **decision support**.

---

## Frontend Dashboard

A React frontend (`credit-ui/`) provides:

- Interactive risk analysis by client ID
- Multi-model comparison
- Risk filtering
- Score sorting
- Model-specific explanations
- User-friendly visualization of results

---

## Project Structure

```bash
credit-risk-analysis/
├── main.py
├── requirements.txt
├── credit_risk_dataset.csv
├── dataset_with_id.csv
├── train_*.py
├── *.pkl
└── credit-ui/
    ├── src/
    ├── public/
    └── package.json
```

---

## Prerequisites

Make sure you have installed:

- Python 3.10+
- Node.js + npm

---

## Backend Setup

### 1. Clone the repository

```bash
git clone https://github.com/lallamimi/credit-risk-analysis.git
cd credit-risk-analysis
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the backend API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Setup

Open another terminal and run:

```bash
cd credit-ui
npm install
npm start
```

## How to Test the Project

### Backend

Once the backend is running, verify that:
- the API starts correctly
- model files are loaded correctly
- requests return predictions without errors

Note:
This project currently relies on the three strongest and most relevant trained models available in the repository:

- XGBoost
- Random Forest
- LightGBM

Some additional model files may be referenced in the backend, but they are not required for the current version of the application.

### Frontend

Once the frontend is running, test:
- entering a client ID
- clicking **Analyser**
- filtering by risk
- selecting a model
- sorting by score
- opening prediction details
- viewing SHAP explanations

---

## Notes

- The Python virtual environment (`.venv`) is required for backend execution and dependency isolation.
- The frontend does **not** require the Python virtual environment.

---

## Key Highlights

- End-to-end machine learning pipeline
- Comparison of several classification models
- Financial risk perspective with Expected Loss
- Interactive React dashboard
- Explainability-oriented frontend
- Combination of backend and frontend components

---

## Future Improvements

- Hyperparameter tuning
- Better model calibration
- Cleaner model loading strategy
- Docker support
- Production deployment
- Extended explainability and reporting

---

## Author

**Oumayma Mektane**
