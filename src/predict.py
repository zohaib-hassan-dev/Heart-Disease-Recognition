"""
src/predict.py
Load saved model and run a single-sample prediction.
Usage:
    python src/predict.py --model results/models/best_model_rf.pkl --input '{"age":54,"sex":1,...}'
"""
import argparse
import json
import joblib
import pandas as pd
from pathlib import Path

def parse_input(input_str: str):
    """
    Accepts JSON string (single sample). Returns DataFrame with one row.
    """
    try:
        sample = json.loads(input_str)
    except Exception as e:
        raise ValueError("Input must be a JSON string representing a dict of features.") from e
    # Ensure it's a 1-row DataFrame
    return pd.DataFrame([sample])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/models/best_model_rf.pkl")
    parser.add_argument("--input", type=str, required=True,
                        help='JSON string of input features, e.g. \'{"age":54,"sex":1,...}\'')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train model first with src/train.py")

    model = joblib.load(model_path)
    X_sample = parse_input(args.input)

    # Match columns expected by model: take intersection & warn about missing
    # The pipeline's preprocessor expects the same columns used during training
    # If feature mismatch occurs, sklearn will error — keep this script simple
    pred = model.predict(X_sample)
    proba = model.predict_proba(X_sample)[:, 1]
    print("Prediction (0=no disease, 1=disease):", int(pred[0]))
    print("Probability of disease:", float(proba[0]))

if __name__ == "__main__":
    main()
