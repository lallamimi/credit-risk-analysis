import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("credit_risk_dataset.csv")
df.columns = df.columns.str.strip()

if "client_id" not in df.columns:
    df["client_id"] = range(1, len(df) + 1)

target = "loan_status"

# =========================
# CLEANING
# =========================
mask_bad = (
    (df["person_age"] < 18) |
    (df["person_age"] > 100) |
    (df["person_emp_length"] > 60)
)
df = df.loc[~mask_bad].copy()
df[target] = df[target].astype(int)

q1, q99 = df["loan_percent_income"].quantile([0.01, 0.99])
df["loan_percent_income"] = df["loan_percent_income"].clip(q1, q99)

# =========================
# FEATURE ENGINEERING
# =========================
def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    income_safe       = data["person_income"].replace(0, np.nan)
    age_working_years = (data["person_age"] - 18).replace(0, np.nan)
    data["loan_to_income"]      = data["loan_amnt"] / income_safe
    data["interest_burden"]     = data["loan_int_rate"] * data["loan_percent_income"]
    data["emp_stability_ratio"] = data["person_emp_length"] / age_working_years
    data["credit_age_ratio"]    = data["cb_person_cred_hist_length"] / data["person_age"].replace(0, np.nan)
    data["rate_x_credit_age"]   = data["loan_int_rate"] / (1 + data["cb_person_cred_hist_length"])
    data["financial_pressure"]  = data["loan_percent_income"] * data["loan_to_income"]
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    data["loan_grade_ord"] = data["loan_grade"].map(grade_map)
    return data

df = add_engineered_features(df)
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# SPLIT
# =========================
X = df.drop(columns=[target, "client_id"])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("Train:", X_train.shape, "  Test:", X_test.shape)

# =========================
# PREPROCESSING
# =========================
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe,     num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

# =========================
# MODEL
# =========================
base = DecisionTreeClassifier(class_weight="balanced", random_state=42)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf",  BaggingClassifier(
        estimator=base,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Bagging Classifier ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")

# =========================
# SAVE
# =========================
pickle.dump(model, open("bagging_model.pkl", "wb"))
print("\nSaved: bagging_model.pkl")
