import pandas as pd
import joblib
from utils.common import MODELS_DIR
from preprocessing.url_lexical_preprocess import extract_url_features
from self_learning.log_prediction import log_url_prediction
model = joblib.load(MODELS_DIR / "url_lexical_model.pkl")
expected_features = joblib.load(MODELS_DIR / "url_lexical_features.pkl")


def predict_url_raw(url: str) -> dict:
    features = extract_url_features(url)
    if features["has_fake_brand_subdomain"] == 1:
     return {
        "source_type": "url",
        "url": url,
        "prediction": "malicious",
        "is_attack": True,
        "attack_type": "phishing_fake_domain",
        "confidence": 0.95,
        "security_score": 10,
        "security_level": "High Risk",
        "features_used": features
    }
    X = pd.DataFrame([features])

    for col in expected_features:
        if col not in X.columns:
            X[col] = 0

    X = X[expected_features]

    proba = float(model.predict_proba(X)[0][1])
    pred = 1 if proba >= 0.5 else 0

    if proba >= 0.75:
        level = "High Risk"
    elif proba >= 0.4:
        level = "Medium Risk"
    else:
        level = "Low Risk"

    security_score = max(0, int((1 - proba) * 100))
    log_url_prediction(url, "malicious" if pred == 1 else "safe", proba)
    return {
        "source_type": "url",
        "url": url,
        "prediction": "malicious" if pred == 1 else "safe",
        "is_attack": bool(pred == 1),
        "attack_type": "phishing_or_malicious_url" if pred == 1 else None,
        "confidence": round(proba, 4),
        "security_score": security_score,
        "security_level": level,
        "features_used": features
    }