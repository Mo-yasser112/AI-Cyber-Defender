import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils.common import BASE_DIR, MODELS_DIR
from utils.dataset_helpers import read_any_file, build_text_from_row

RAW_DIR = BASE_DIR / "data" / "web_raw"
MODEL_PATH = MODELS_DIR / "web_model.pkl"
FEATURES_PATH = MODELS_DIR / "web_features.pkl"
LABEL_CANDIDATES = ["label", "Label", "class", "Class", "target", "Target", "attack", "Attack"]

def detect_label_column(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("Could not detect label column in web dataset.")

def normalize_label(v):
    s = str(v).strip().lower()
    return 0 if s in {"0", "normal", "benign", "legitimate", "safe"} else 1

def main():
    files = [p for p in RAW_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl", ".ndjson"}]
    if not files:
        raise FileNotFoundError(f"No supported web data files found in {RAW_DIR}")

    frames = []
    for file in files:
        print("Reading:", file.name)
        df_part = read_any_file(file)
        if not df_part.empty:
            frames.append(df_part)

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    label_col = detect_label_column(df)

    df = df.dropna(subset=[label_col]).copy()
    df["label_bin"] = df[label_col].apply(normalize_label)
    df["text"] = df.apply(lambda row: build_text_from_row(row, exclude_cols={label_col, "label_bin"}), axis=1)
    df = df[df["text"].str.len() > 0].copy()

    # لو الداتا كلها attack فقط، نضيف normal samples صناعية
    if df["label_bin"].nunique() == 1:
        print("Only one class detected. Adding synthetic normal samples...")

        normal_samples = pd.DataFrame({
            "text": [
                "method=GET path=/index.html status=200",
                "method=GET path=/home status=200",
                "method=GET path=/about status=200",
                "method=GET path=/contact status=200",
                "method=POST path=/login username=user status=200",
                "method=GET path=/products?id=10 status=200",
                "method=GET path=/dashboard status=200",
                "method=POST path=/search query=phone status=200",
                "method=GET path=/api/user/profile status=200",
                "method=GET path=/services status=200"
            ] * 1000,
            "label_bin": [0] * 10000
        })

        df = pd.concat([df[["text", "label_bin"]], normal_samples], ignore_index=True)

    X = df["text"]
    y = df["label_bin"]

    print("Web dataset shape:", df.shape)
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"type": "text", "label_column": label_col}, FEATURES_PATH)
    print("\nSaved:", MODEL_PATH.name, FEATURES_PATH.name)

if __name__ == "__main__":
    main()