import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils.common import BASE_DIR, MODELS_DIR
from utils.model_io import save_model_bundle
from preprocessing.url_lexical_preprocess import preprocess_url_dataframe

DATA_PATH = BASE_DIR / "data" / "url" / "url_raw_dataset.csv"
MODEL_PATH = MODELS_DIR / "url_lexical_model.pkl"
FEATURES_PATH = MODELS_DIR / "url_lexical_features.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'url' and 'label' columns.")

    df = df.dropna(subset=["url", "label"]).copy()
    df["label"] = df["label"].astype(int)

    processed_df = preprocess_url_dataframe(df)

    X = processed_df.drop(columns=["label"])
    y = processed_df["label"]

    print("Processed shape:", X.shape)
    print("Label distribution:")
    print(y.value_counts())

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

    print("\\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\\nConfusion Matrix:\\n", confusion_matrix(y_test, y_pred))
    print("\\nClassification Report:\\n", classification_report(y_test, y_pred))

    save_model_bundle(model, X.columns, MODEL_PATH, FEATURES_PATH)
    print("\\nSaved:", MODEL_PATH.name, FEATURES_PATH.name)


if __name__ == "__main__":
    main()