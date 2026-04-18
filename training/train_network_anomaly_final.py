import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils.common import BASE_DIR, MODELS_DIR
from utils.model_io import save_model_bundle

DATA_PATH = BASE_DIR / "data" / "network_processed" / "cicids_full.csv"
MODEL_PATH = MODELS_DIR / "network_anomaly_model.pkl"
FEATURES_PATH = MODELS_DIR / "network_features.pkl"
DROP_COLS = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]

def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("Expected 'Label' column in CICIDS dataset.")

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["Label"] = df["Label"].astype(str).str.strip().apply(lambda x: 0 if x == "BENIGN" else 1)

    X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
    valid_idx = X.dropna().index
    X = X.loc[valid_idx]
    y = df.loc[valid_idx, "Label"]

    print("Network anomaly processed shape:", X.shape)
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    save_model_bundle(model, X.columns, MODEL_PATH, FEATURES_PATH)
    print("\nSaved:", MODEL_PATH.name, FEATURES_PATH.name)

if __name__ == "__main__":
    main()
