from utils.common import MODELS_DIR
from utils.model_io import load_model, load_features

url_model = load_model(MODELS_DIR / "url_lexical_model.pkl")
url_features = load_features(MODELS_DIR / "url_features.pkl")

network_anomaly_model = load_model(MODELS_DIR / "network_anomaly_model.pkl")
network_features = load_features(MODELS_DIR / "network_features.pkl")

network_attack_model = load_model(MODELS_DIR / "network_attack_classifier.pkl")
network_classifier_features = load_features(MODELS_DIR / "network_classifier_features.pkl")

web_model = load_model(MODELS_DIR / "web_model.pkl")
web_features = load_features(MODELS_DIR / "web_features.pkl")

windows_model = load_model(MODELS_DIR / "windows_model.pkl")
windows_features = load_features(MODELS_DIR / "windows_features.pkl")
