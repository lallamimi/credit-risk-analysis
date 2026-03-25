# Credit Risk Analysis & Prediction

<p align="center">
  <b>Machine Learning project for credit default prediction with real financial impact</b><br>
  <i>From data preprocessing to business-driven risk estimation</i>
</p>

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost, LightGBM
* Matplotlib
* React (Frontend)

---

## Overview

This project focuses on predicting the probability of loan default using machine learning techniques.

Unlike standard ML projects, it introduces a **business perspective** by estimating:

> **Expected Loss (EL) = PD × LGD × EAD**

This allows transforming predictions into **real financial risk**, which is essential in banking and fintech.

---

## Objectives

* Predict loan default risk
* Handle imbalanced datasets
* Build a complete ML pipeline
* Compare multiple models
* Translate predictions into business insights

---

## Dataset

* Credit Risk Dataset
* ~32,000+ observations
* Target: `loan_status`

### Key features:

* Age, income
* Employment length
* Loan amount & interest rate
* Credit history

---

## Machine Learning Pipeline

### Data Preprocessing

* Missing values handling
* One-Hot Encoding
* Feature scaling

### Feature Engineering

* Financial ratios (loan-to-income, etc.)
* Derived predictive features

### Models Implemented

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* AdaBoost / Bagging
* XGBoost
* LightGBM

---

## Model Evaluation

Due to class imbalance:

**F1-score is the main metric**

Other metrics:

* Accuracy
* Precision / Recall
* Confusion Matrix

Special focus on **recall** to minimize financial risk.

---

## Business Impact

This project connects machine learning with real-world finance:

* Identify high-risk borrowers
* Estimate potential financial losses
* Support credit decision systems

Moves from **prediction → decision-making**

---

## Frontend Dashboard

A React-based UI (`credit-ui/`) allows:

* Data visualization
* Interactive exploration
* Understanding model outputs

---

##  How to Run

### 1. Install dependencies

```bash id="9r5i1d"
pip install -r requirements.txt
```

### 2. Train models

```bash id="v6h3qj"
python train_randomforest.py
```

### 3. Run project

```bash id="4o6lq6"
python main.py
```

### 4. Launch frontend

```bash id="x3q6k2"
cd credit-ui
npm install
npm start
```

---

## Key Highlights

End-to-end ML pipeline
Multiple models comparison
Imbalanced data handling
Business-driven metrics (Expected Loss)
Full stack (ML + Frontend)

---

## Future Improvements

* SHAP (model explainability)
* Hyperparameter tuning
* API deployment (FastAPI)
* Real-time predictions

---

## Author

**Oumayma Mektane**

---

