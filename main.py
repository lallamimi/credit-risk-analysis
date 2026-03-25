import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pickle
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. SHAP explanations will be unavailable.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
def try_load(path):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        print(f"Warning: {path} not found — model will show N/A in app.")
        return None

logistic_model = pickle.load(open("logreg_model.pkl",   "rb"))
xgb_model      = pickle.load(open("xgboost_model.pkl",  "rb"))
rf_model       = try_load("randomforest_model.pkl")
gb_model       = try_load("gradientboosting_model.pkl")
lgbm_model     = try_load("lightgbm_model.pkl")
dt_model       = try_load("decisiontree_model.pkl")
ada_model      = try_load("adaboost_model.pkl")
et_model       = try_load("extratrees_model.pkl")
bag_model      = try_load("bagging_model.pkl")

df = pd.read_csv("dataset_with_id.csv")

# --------------------------------------------------
# PRE-CACHE SHAP EXPLAINERS  (done once at startup)
# --------------------------------------------------
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.ensemble import BaggingClassifier as _Bag, AdaBoostClassifier as _Ada

_NO_TREE_SHAP = (_LR, _Bag, _Ada)

SHAP_EXPLAINERS: dict = {}

def _init_shap_explainers():
    if not SHAP_AVAILABLE:
        return
    _candidates = {
        "logistic":      logistic_model,
        "xgboost":       xgb_model,
        "randomforest":  rf_model,
        "gradientboost": gb_model,
        "lightgbm":      lgbm_model,
        "decisiontree":  dt_model,
        "adaboost":      ada_model,
        "extratrees":    et_model,
        "bagging":       bag_model,
    }
    for k, mdl in _candidates.items():
        if mdl is None:
            continue
        clf = mdl.named_steps["clf"]
        if isinstance(clf, _NO_TREE_SHAP):
            continue
        try:
            SHAP_EXPLAINERS[k] = shap.TreeExplainer(clf)
            print(f"SHAP ready: {k}")
        except Exception as e:
            print(f"SHAP init failed for {k}: {e}")

_init_shap_explainers()

# --------------------------------------------------
# HUMAN-READABLE FEATURE NAMES
# --------------------------------------------------
def interpret_feature(name, contribution):
    name = name.replace("num__", "").replace("cat__", "")

    if name == "person_age":
        return "Âge du demandeur"
    if name == "person_income":
        return "Revenu faible" if contribution > 0 else "Revenu élevé"
    if name == "person_emp_length":
        return "Ancienneté professionnelle courte" if contribution > 0 else "Ancienneté professionnelle"
    if name == "loan_amnt":
        return "Montant du prêt élevé" if contribution > 0 else "Montant du prêt modéré"
    if name == "loan_int_rate":
        return "Taux d'intérêt élevé" if contribution > 0 else "Taux d'intérêt faible"
    if name == "loan_percent_income":
        return "Taux d'endettement élevé" if contribution > 0 else "Taux d'endettement maîtrisé"
    if name == "cb_person_cred_hist_length":
        return "Historique de crédit court" if contribution > 0 else "Historique de crédit long"
    if name.startswith("loan_grade_ord"):
        return "Mauvaise notation du prêt" if contribution > 0 else "Bonne notation du prêt"
    if name.startswith("loan_grade_"):
        grade = name.split("_")[-1]
        return f"Notation du prêt : {grade}"
    if name.startswith("person_home_ownership_"):
        status = name.split("_")[-1]
        return f"Statut résidentiel : {status}"
    if name.startswith("loan_intent_"):
        intent = name.replace("loan_intent_", "")
        return f"Objectif du prêt : {intent}"
    if name.startswith("cb_person_default_on_file_"):
        val = name.split("_")[-1]
        return "Présence d'un défaut de paiement" if val == "Y" else "Aucun défaut antérieur"
    if name == "loan_to_income":
        return "Ratio prêt/revenu élevé" if contribution > 0 else "Ratio prêt/revenu faible"
    if name == "interest_burden":
        return "Charge d'intérêts élevée" if contribution > 0 else "Charge d'intérêts faible"
    if name == "emp_stability_ratio":
        return "Instabilité professionnelle" if contribution > 0 else "Stabilité professionnelle"
    if name == "credit_age_ratio":
        return "Historique de crédit insuffisant" if contribution > 0 else "Historique de crédit solide"
    if name == "rate_x_credit_age":
        return "Taux élevé avec crédit récent" if contribution > 0 else "Taux maîtrisé"
    if name == "financial_pressure":
        return "Pression financière élevée" if contribution > 0 else "Pression financière maîtrisée"

    return name


# --------------------------------------------------
# SUGGESTIONS
# --------------------------------------------------
def generate_suggestions(risk_factors):
    suggestions = []
    for factor in risk_factors:
        if "Taux d'endettement élevé" in factor:
            suggestions.append("Réduire le taux d'endettement")
        if "Revenu faible" in factor:
            suggestions.append("Augmenter les revenus déclarés")
        if "Montant du prêt élevé" in factor or "Ratio prêt/revenu élevé" in factor:
            suggestions.append("Demander un montant de prêt plus faible")
        if "Taux d'intérêt élevé" in factor or "Charge d'intérêts élevée" in factor or "Taux élevé" in factor:
            suggestions.append("Améliorer le profil pour obtenir un meilleur taux")
        if "défaut" in factor:
            suggestions.append("Améliorer l'historique de crédit")
        if "Instabilité professionnelle" in factor:
            suggestions.append("Consolider la stabilité professionnelle")
        if "Historique de crédit insuffisant" in factor or "Historique de crédit court" in factor:
            suggestions.append("Développer l'historique de crédit")
        if "Pression financière élevée" in factor:
            suggestions.append("Réduire les engagements financiers existants")
    return list(dict.fromkeys(suggestions))[:3]


# --------------------------------------------------
# SHAP EXPLAINER  (uses pre-cached explainers)
# --------------------------------------------------
def get_shap_values(model_key: str, pipeline, X_client):
    if not SHAP_AVAILABLE:
        return []
    try:
        preprocess    = pipeline.named_steps["prep"]
        clf           = pipeline.named_steps["clf"]
        X_transformed = preprocess.transform(X_client)
        feature_names = list(preprocess.get_feature_names_out())

        if isinstance(clf, _LR):
            vals = X_transformed[0] * clf.coef_[0]
        elif model_key not in SHAP_EXPLAINERS:
            return []
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = SHAP_EXPLAINERS[model_key].shap_values(X_transformed)
            if isinstance(raw, list):
                vals = raw[1][0]
            elif len(raw.shape) == 3:
                vals = raw[0, :, 1]
            else:
                vals = raw[0]

        result = [
            {"feature": interpret_feature(name, float(v)), "shap_value": round(float(v), 4)}
            for name, v in zip(feature_names, vals)
        ]
        result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return result[:10]
    except Exception as e:
        print(f"SHAP error ({model_key}): {e}")
        return []


# --------------------------------------------------
# CONTRIBUTION FALLBACK (when SHAP unavailable/fails)
# --------------------------------------------------
def get_contributions_fallback(pipeline, X_client):
    prep   = pipeline.named_steps["prep"]
    clf    = pipeline.named_steps["clf"]
    Xt     = prep.transform(X_client)
    fnames = list(prep.get_feature_names_out())

    if isinstance(clf, _LR):
        contribs = list(Xt[0] * clf.coef_[0])
    elif isinstance(clf, _Bag):
        imps = np.array([
            est.feature_importances_ for est in clf.estimators_
            if hasattr(est, "feature_importances_")
        ])
        importances = imps.mean(axis=0) if len(imps) else np.ones(len(fnames)) / len(fnames)
        contribs = [float(imp) * (1 if float(Xt[0][i]) > 0 else -1)
                    for i, imp in enumerate(importances)]
    elif hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        contribs = [float(imp) * (1 if float(Xt[0][i]) > 0 else -1)
                    for i, imp in enumerate(importances)]
    else:
        contribs = [0.0] * len(fnames)

    edf = pd.DataFrame({"feature": fnames, "contribution": contribs}) \
            .sort_values("contribution", ascending=False)

    facteurs_risque = [
        interpret_feature(r["feature"], r["contribution"])
        for _, r in edf.head(3).iterrows()
    ]
    facteurs_protecteurs = [
        interpret_feature(r["feature"], r["contribution"])
        for _, r in edf.tail(3).iterrows()
    ]

    shap_like = [
        {
            "feature":    interpret_feature(r["feature"], r["contribution"]),
            "shap_value": round(float(r["contribution"]), 4),
        }
        for _, r in edf.iterrows()
        if r["contribution"] != 0
    ]
    shap_like.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    shap_like = shap_like[:10]

    return facteurs_risque, facteurs_protecteurs, shap_like


# --------------------------------------------------
# RISK LEVEL HELPER
# --------------------------------------------------
def risk_level(score_pct: float) -> str:
    if score_pct > 60:
        return "high"
    if score_pct > 30:
        return "medium"
    return "low"


# --------------------------------------------------
# COMBINED ENDPOINT — one call, all models in parallel
# --------------------------------------------------
_ALL_MODELS = {
    "logistic":      None,
    "xgboost":       None,
    "randomforest":  None,
    "gradientboost": None,
    "lightgbm":      None,
    "decisiontree":  None,
    "adaboost":      None,
    "extratrees":    None,
    "bagging":       None,
}

def _get_all_models():
    return {
        "logistic":      logistic_model,
        "xgboost":       xgb_model,
        "randomforest":  rf_model,
        "gradientboost": gb_model,
        "lightgbm":      lgbm_model,
        "decisiontree":  dt_model,
        "adaboost":      ada_model,
        "extratrees":    et_model,
        "bagging":       bag_model,
    }

def _run_one_model(key, mdl, X_client):
    """Run inference + factor extraction for a single model."""
    if mdl is None:
        return key, {"score": None, "risk": "unknown",
                     "facteurs_risque": [], "facteurs_protecteurs": [], "recommandations": []}
    try:
        score = round(float(mdl.predict_proba(X_client)[0][1]) * 100, 2)
        level = risk_level(score)
        fr, fp, _ = get_contributions_fallback(mdl, X_client)
        return key, {
            "score":                score,
            "risk":                 level,
            "facteurs_risque":      fr,
            "facteurs_protecteurs": fp,
            "recommandations":      generate_suggestions(fr),
        }
    except Exception as e:
        print(f"Model error ({key}): {e}")
        return key, {"score": None, "risk": "unknown",
                     "facteurs_risque": [], "facteurs_protecteurs": [], "recommandations": []}


@app.get("/predict_all/{client_id}")
def predict_all(client_id: int):
    client = df[df["client_id"] == client_id]
    if client.empty:
        return {"error": "Client not found"}

    X_client = client.drop(columns=["loan_status"])
    named_models = _get_all_models()

    with ThreadPoolExecutor() as ex:
        futures = {ex.submit(_run_one_model, k, m, X_client): k
                   for k, m in named_models.items()}
        models_result = {f.result()[0]: f.result()[1] for f in futures}

    available_scores = [v["score"] for v in models_result.values() if v["score"] is not None]
    avg_score  = sum(available_scores) / len(available_scores) if available_scores else 50
    final_risk = risk_level(avg_score)

    lr = models_result.get("logistic", {})
    lr_proba = lr.get("score", 0) / 100 if lr.get("score") is not None else 0
    card = {
        "client_id":            client_id,
        "solvable":             bool(lr_proba < 0.5),
        "score_percent":        lr.get("score", 0),
        "facteurs_risque":      lr.get("facteurs_risque", []),
        "facteurs_protecteurs": lr.get("facteurs_protecteurs", []),
        "recommandations":      lr.get("recommandations", []),
    }

    return {
        "card":       card,
        "client_id":  client_id,
        "models":     models_result,
        "final_risk": final_risk,
        "avg_score":  round(avg_score, 2),
    }


# --------------------------------------------------
# ON-DEMAND SHAP ENDPOINT
# --------------------------------------------------
@app.get("/shap/{client_id}/{model_key}")
def shap_for_model(client_id: int, model_key: str):
    _model_map = {
        "logistic":      logistic_model,
        "xgboost":       xgb_model,
        "randomforest":  rf_model,
        "gradientboost": gb_model,
        "lightgbm":      lgbm_model,
        "decisiontree":  dt_model,
        "adaboost":      ada_model,
        "extratrees":    et_model,
        "bagging":       bag_model,
    }

    mdl = _model_map.get(model_key)
    if mdl is None:
        return {"shap": [], "error": "Model not found or not loaded"}

    client = df[df["client_id"] == client_id]
    if client.empty:
        return {"shap": [], "error": "Client not found"}

    X_client = client.drop(columns=["loan_status"])

    shap_vals = get_shap_values(model_key, mdl, X_client)
    if not shap_vals:
        try:
            _, _, shap_vals = get_contributions_fallback(mdl, X_client)
        except Exception as e:
            print(f"SHAP fallback failed for {model_key}: {e}")
            shap_vals = []

    return {"shap": shap_vals, "model_key": model_key, "client_id": client_id}
