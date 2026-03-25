# Credit Risk Analysis & Prediction

<p align="center">
  <b>Machine Learning project for credit default prediction with real financial impact</b><br>
  <i>From data preprocessing to business-driven risk estimation</i>
</p>

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost, LightGBM  
- Matplotlib  
- React (Frontend)  

---

## Overview

This project focuses on predicting the probability of loan default using machine learning techniques.

Unlike standard ML projects, it introduces a business perspective by estimating:

> Expected Loss (EL) = PD × LGD × EAD

- PD (Probability of Default): predicted by the model  
- LGD (Loss Given Default): proportion of loss if default occurs  
- EAD (Exposure at Default): total loan exposure  

This approach allows translating model predictions into real financial risk, which is essential in banking and fintech.

---

## Objectives

- Predict loan default risk  
- Handle imbalanced datasets  
- Build a complete machine learning pipeline  
- Compare multiple models  
- Translate predictions into business insights  

---

## Dataset

- Credit Risk Dataset  
- Approximately 32,000 observations  
- Target variable: `loan_status`  

### Key features

- Age, income  
- Employment length  
- Loan amount and interest rate  
- Credit history  

---

## Machine Learning Pipeline

### Data Preprocessing

- Handling missing values  
- Encoding categorical variables (One-Hot Encoding)  
- Feature scaling  

### Feature Engineering

- Financial ratios (e.g. loan-to-income)  
- Derived predictive features  

### Models Implemented

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- AdaBoost / Bagging  
- XGBoost  
- LightGBM  

---

## Model Evaluation

Due to class imbalance, F1-score is used as the main evaluation metric.

Additional metrics:
- Accuracy  
- Precision  
- Recall  
- Confusion Matrix  

Special attention is given to recall in order to minimize financial risk.

Example result:
- F1-score: ~0.78 (Random Forest)

---

## Business Impact

This project connects machine learning with real-world financial decision-making:

- Identify high-risk borrowers  
- Estimate potential financial losses  
- Support credit approval decisions  

It shifts the focus from prediction to decision support.

---

## Frontend Dashboard

A React-based interface (`credit-ui/`) provides:

- Data visualization  
- Interactive exploration  
- Insight into model behavior  

---

## Quick Start

```bash
git clone https://github.com/your-username/credit-risk-analysis.git
cd credit-risk-analysis
pip install -r requirements.txt
python main.py
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a model

```bash
python train_randomforest.py
```

### 3. Run the main script

```bash
python main.py
```

### 4. Launch the frontend

```bash
cd credit-ui
npm install
npm start
```

---

## Key Highlights

- End-to-end machine learning pipeline  
- Comparison of multiple models  
- Handling of imbalanced data  
- Integration of business metrics (Expected Loss)  
- Combination of backend (Python) and frontend (React)  

---

## Future Improvements

- Model interpretability (SHAP)  
- Hyperparameter tuning  
- API deployment (FastAPI or Flask)  
- Real-time prediction system  

---

## Author

Oumayma Mektane   

---

