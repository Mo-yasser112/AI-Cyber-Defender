import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, accuracy_score

from preprocessing.url_lexical_preprocess import preprocess_url_dataframe

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "url"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "self_learning" / "logs"
VERSIONS_DIR = MODELS_DIR / "versions"

BASE_DATASET_PATH = DATA_DIR / "url_raw_dataset.csv"
INCOMING_LOG_PATH = LOGS_DIR / "url_predictions_log.csv"
MODEL_PATH = MODELS_DIR / "url_lexical_model.pkl"
FEATURES_PATH = MODELS_DIR / "url_lexical_features.pkl"

VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.95
MIN_NEW_SAMPLES = 20


def load_base_dataset() -> pd.DataFrame:
    df = pd.read_csv(BASE_DATASET_PATH)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["url", "label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def load_logged_predictions() -> pd.DataFrame:
    if not INCOMING_LOG_PATH.exists():
        return pd.DataFrame(columns=["url", "prediction", "confidence", "timestamp"])

    df = pd.read_csv(INCOMING_LOG_PATH)
    if df.empty:
        return df

    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["url", "prediction", "confidence"]).copy()
    return df


def select_high_confidence_samples(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return pd.DataFrame(columns=["url", "label"])

    selected = log_df[log_df["confidence"] >= CONFIDENCE_THRESHOLD].copy()
    if selected.empty:
        return pd.DataFrame(columns=["url", "label"])

    selected["label"] = selected["prediction"].map({"safe": 0, "malicious": 1})
    selected = selected.dropna(subset=["label"])
    selected["label"] = selected["label"].astype(int)

    return selected[["url", "label"]].drop_duplicates(subset=["url"])


def merge_datasets(base_df: pd.DataFrame, pseudo_df: pd.DataFrame) -> pd.DataFrame:
    if pseudo_df.empty:
        return base_df.copy()

    combined = pd.concat([base_df, pseudo_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"], keep="last")
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined


def train_candidate_model(train_df: pd.DataFrame):
    processed_df = preprocess_url_dataframe(train_df)

    X = processed_df.drop(columns=["label"])
    y = processed_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }

    return model, list(X.columns), metrics


def evaluate_current_model(base_df: pd.DataFrame):
    processed_df = preprocess_url_dataframe(base_df)

    X = processed_df.drop(columns=["label"])
    y = processed_df["label"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    current_model = joblib.load(MODEL_PATH)
    y_pred = current_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred)
    }
    return metrics


def save_new_version(model, features, metrics):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_path = VERSIONS_DIR / f"url_lexical_model_{timestamp}.pkl"
    features_version_path = VERSIONS_DIR / f"url_lexical_features_{timestamp}.pkl"

    joblib.dump(model, model_version_path)
    joblib.dump(features, features_version_path)

   
    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, FEATURES_PATH)

    meta = {
        "timestamp": timestamp,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"]
    }

    with open(VERSIONS_DIR / f"url_lexical_model_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] New model version saved: {model_version_path.name}")
    print(f"[INFO] New model promoted to active model.")


def main():
    print("[INFO] Loading base dataset...")
    base_df = load_base_dataset()

    print("[INFO] Loading logged predictions...")
    log_df = load_logged_predictions()

    pseudo_df = select_high_confidence_samples(log_df)
    print(f"[INFO] High-confidence pseudo-labeled samples: {len(pseudo_df)}")

    if len(pseudo_df) < MIN_NEW_SAMPLES:
        print("[INFO] Not enough new high-confidence samples. Skipping retraining.")
        return

    train_df = merge_datasets(base_df, pseudo_df)
    print(f"[INFO] Final training dataset size: {len(train_df)}")

    print("[INFO] Evaluating current model...")
    current_metrics = evaluate_current_model(base_df)
    print(f"[INFO] Current model F1: {current_metrics['f1']:.4f}")

    print("[INFO] Training candidate model...")
    candidate_model, candidate_features, candidate_metrics = train_candidate_model(train_df)
    print(f"[INFO] Candidate model F1: {candidate_metrics['f1']:.4f}")
    print(candidate_metrics["report"])

    if candidate_metrics["f1"] > current_metrics["f1"]:
        print("[INFO] Candidate model is better. Promoting new model...")
        save_new_version(candidate_model, candidate_features, candidate_metrics)
    else:
        print("[INFO] Candidate model is not better. Keeping current model.")


if __name__ == "__main__":
    main()