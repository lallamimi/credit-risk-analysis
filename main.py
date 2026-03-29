import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pickle
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# OPTIONAL SHAP
# --------------------------------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. SHAP explanations unavailable.")

# --------------------------------------------------
# FASTAPI INIT
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# SAFE MODEL LOADING
# --------------------------------------------------
def try_load(path):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        print(f"Warning: {path} not found.")
        return None

# Models
xgb_model = try_load("xgboost_model.pkl")
lgbm_model = try_load("lightgbm_model.pkl")
rf_model = try_load("randomforest_model.pkl")

df = pd.read_csv("dataset_with_id.csv")

# --------------------------------------------------
# FEATURE LABELS
# --------------------------------------------------
def interpret_feature(name, contribution):
    name = name.replace("num__", "").replace("cat__", "")

    mapping = {
        "person_age": "Applicant age",
        "loan_amnt": "Loan amount",
        "loan_int_rate": "Interest rate",
        "loan_percent_income": "Debt-to-income ratio",
        "cb_person_cred_hist_length": "Credit history length",
        "loan_to_income": "Loan-to-income ratio",
        "interest_burden": "Interest burden",
        "financial_pressure": "Financial pressure",
        "loan_grade_ord": "Loan grade level",
        "person_income": "Income",
        "person_home_ownership_RENT": "Renter",
        "person_home_ownership_MORTGAGE": "Mortgage holder",
        "person_home_ownership_OWN": "Home owner",
        "loan_intent_VENTURE": "Business venture",
        "loan_intent_HOMEIMPROVEMENT": "Home improvement",
    }

    return mapping.get(name, name)

# --------------------------------------------------
# SUGGESTIONS
# --------------------------------------------------
def generate_suggestions(risk_factors):
    suggestions = []
    lowered = [f.lower() for f in risk_factors]

    if any("debt" in f for f in lowered):
        suggestions.append("Reduce debt-to-income ratio")

    if any("income" in f for f in lowered):
        suggestions.append("Increase income")

    if any("loan" in f for f in lowered):
        suggestions.append("Reduce loan amount")

    return suggestions[:3]

# --------------------------------------------------
# FALLBACK CONTRIBUTIONS
# --------------------------------------------------
def get_contributions_fallback(pipeline, X_client):
    try:
        prep = pipeline.named_steps["prep"]
        clf = pipeline.named_steps["clf"]

        Xt = prep.transform(X_client)
        fnames = list(prep.get_feature_names_out())

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            contribs = [
                float(importances[i]) * (1 if Xt[0][i] > 0 else -1)
                for i in range(len(fnames))
            ]
        else:
            contribs = [0] * len(fnames)

        df_contrib = pd.DataFrame({
            "feature": fnames,
            "contribution": contribs
        }).sort_values("contribution", ascending=False)

        risk = [
            interpret_feature(r["feature"], r["contribution"])
            for _, r in df_contrib.head(3).iterrows()
        ]

        protect = [
            interpret_feature(r["feature"], r["contribution"])
            for _, r in df_contrib.tail(3).iterrows()
        ]

        return risk, protect

    except Exception as e:
        print("Contribution error:", e)
        return [], []

# --------------------------------------------------
# RISK LEVEL
# --------------------------------------------------
def risk_level(score):
    if score > 60:
        return "high"
    elif score > 30:
        return "medium"
    return "low"

# --------------------------------------------------
# MODELS DICT
# --------------------------------------------------
def get_models():
    return {
        "xgboost": xgb_model,
        "lightgbm": lgbm_model,
        "randomforest": rf_model,
    }

# --------------------------------------------------
# MODEL RUN
# --------------------------------------------------
def run_model(name, model, X):
    if model is None:
        return name, {"score": None}

    try:
        score = round(float(model.predict_proba(X)[0][1]) * 100, 2)
        risk, protect = get_contributions_fallback(model, X)

        return name, {
            "score": score,
            "risk": risk_level(score),
            "risk_factors": risk,
            "protective_factors": protect,
            "recommendations": generate_suggestions(risk)
        }

    except Exception as e:
        print(f"Error {name}: {e}")
        return name, {"score": None}

# --------------------------------------------------
# SHAP HELPERS
# --------------------------------------------------
def get_shap_values_for_model(model, X_client):
    if not SHAP_AVAILABLE:
        raise Exception("SHAP is not installed on the backend.")

    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]

    Xt = prep.transform(X_client)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()

    feature_names = list(prep.get_feature_names_out())

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(Xt)

    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            values = shap_values[1][0]
        else:
            values = shap_values[0][0]
    else:
        shap_values = np.array(shap_values)

        if shap_values.ndim == 2:
            values = shap_values[0]
        elif shap_values.ndim == 3:
            if shap_values.shape[2] > 1:
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0, :, 0]
        else:
            raise Exception(f"Unexpected SHAP format: {shap_values.shape}")

    df_shap = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values
    })

    df_shap["abs_val"] = df_shap["shap_value"].abs()
    df_shap = df_shap.sort_values("abs_val", ascending=False).head(12)

    result = [
        {
            "feature": interpret_feature(row["feature"], row["shap_value"]),
            "shap_value": float(row["shap_value"])
        }
        for _, row in df_shap.iterrows()
    ]

    return result

# --------------------------------------------------
# SHAP ENDPOINT
# --------------------------------------------------
@app.get("/shap/{client_id}/{model_name}")
def get_shap(client_id: int, model_name: str):
    if not SHAP_AVAILABLE:
        return {"error": "SHAP is not installed on the server.", "shap": []}

    models = get_models()
    model = models.get(model_name)

    if model is None:
        return {"error": f"Model '{model_name}' not found.", "shap": []}

    client = df[df["client_id"] == client_id]
    if client.empty:
        return {"error": "Client not found", "shap": []}

    try:
        X = client.drop(columns=["loan_status"])
        shap_rows = get_shap_values_for_model(model, X)
        return {
            "client_id": client_id,
            "model": model_name,
            "shap": shap_rows
        }
    except Exception as e:
        print(f"SHAP error ({model_name}): {e}")
        return {"error": f"SHAP error for {model_name}: {str(e)}", "shap": []}

# --------------------------------------------------
# MAIN ENDPOINT
# --------------------------------------------------
@app.get("/predict_all/{client_id}")
def predict_all(client_id: int):
    client = df[df["client_id"] == client_id]
    if client.empty:
        return {"error": "Client not found"}

    X = client.drop(columns=["loan_status"])
    models = get_models()

    with ThreadPoolExecutor() as executor:
        results = dict(executor.map(lambda item: run_model(*item, X), models.items()))

    valid_scores = [m["score"] for m in results.values() if m["score"] is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 50

    ref = results.get("xgboost", {})

    card = {
        "client_id": client_id,
        "score_percent": ref.get("score"),
        "approvable": ref.get("score", 0) < 50,
        "risk_factors": ref.get("risk_factors", []),
        "protective_factors": ref.get("protective_factors", []),
        "recommendations": ref.get("recommendations", [])
    }

    return {
        "card": card,
        "models": results,
        "avg_score": round(avg_score, 2),
        "final_risk": risk_level(avg_score)
    }
