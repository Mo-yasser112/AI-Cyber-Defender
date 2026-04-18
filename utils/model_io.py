import joblib
from pathlib import Path

def save_model_bundle(model, feature_names, output_model_path: Path, output_features_path: Path):
    joblib.dump(model, output_model_path)
    joblib.dump(list(feature_names), output_features_path)

def load_model(path: Path):
    return joblib.load(path)

def load_features(path: Path):
    return joblib.load(path)
