import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "self_learning" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

URL_LOG_PATH = LOGS_DIR / "url_predictions_log.csv"


def log_url_prediction(url: str, prediction: str, confidence: float):
    row = {
        "url": url,
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }

    if URL_LOG_PATH.exists():
        df = pd.read_csv(URL_LOG_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(URL_LOG_PATH, index=False)