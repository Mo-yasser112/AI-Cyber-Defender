from preprocessing.text_preprocess import preprocess_text_input
from inference.model_loader import web_model

def predict_web(data: dict) -> dict:
    X = preprocess_text_input(data)
    pred = int(web_model.predict(X)[0])
    return {
        "source_type": "web",
        "prediction": "attack" if pred == 1 else "normal",
        "is_attack": bool(pred == 1),
        "attack_type": "web_attack" if pred == 1 else None
    }
