"""
src/data_loader.py
Helpers to load and save raw / processed data
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_raw(path: str = None) -> pd.DataFrame:
    """
    Load raw dataset. If path is None, expects data/raw/heart.csv under project root.
    """
    if path is None:
        # Try a couple of common raw filenames that may be present in this repo
        candidates = [PROJECT_ROOT / "data" / "raw" / "heart.csv",
                      PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv",
                      PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"]
        for p in candidates:
            if p.exists():
                return pd.read_csv(p)
        # none found, raise clear error
        raise FileNotFoundError(
            f"Raw data file not found. Expected one of: {[str(p.name) for p in candidates]} in {PROJECT_ROOT / 'data' / 'raw'}"
        )
    return pd.read_csv(path)

def save_processed(df: pd.DataFrame, filename: str = "heart_cleaned.csv"):
    """
    Save processed dataframe to data/processed/
    """
    out = PROJECT_ROOT / "data" / "processed" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved processed data to: {out}")

def load_processed(path: str = None) -> pd.DataFrame:
    """
    Load processed dataset. If path is None, expects data/processed/heart_cleaned.csv
    """
    if path is None:
        path = PROJECT_ROOT / "data" / "processed" / "heart_cleaned.csv"
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Processed file at {path} is empty. Remove it to force preprocessing or replace with a valid processed CSV.")
    except FileNotFoundError:
        raise
    return df

if __name__ == "__main__":
    # quick test
    try:
        df = load_raw()
        print("Raw data loaded. Shape:", df.shape)
    except Exception as e:
        print("Could not load raw data. Put your CSV at data/raw/heart.csv")
