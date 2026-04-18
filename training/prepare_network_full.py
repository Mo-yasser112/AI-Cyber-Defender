import pandas as pd
from utils.common import BASE_DIR

RAW_DIR = BASE_DIR / "data" / "network_raw"
OUT_PATH = BASE_DIR / "data" / "network_processed" / "cicids_full.csv"

def main():
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CICIDS csv files found in {RAW_DIR}")

    frames = []
    for file in csv_files:
        print(f"Reading: {file.name}")
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    full_df.columns = full_df.columns.str.strip()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_PATH, index=False)
    print("Saved merged dataset:", OUT_PATH)
    print("Final shape:", full_df.shape)

if __name__ == "__main__":
    main()
