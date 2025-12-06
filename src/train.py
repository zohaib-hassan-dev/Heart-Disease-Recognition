"""
src/train.py
Train models, do CV, save best model and the preprocessor.
Usage:
    python src/train.py --model rf --print-cv
    python src/train.py --model lr --raw-path data/raw/custom_heart.csv
"""
import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from data_loader import load_processed, save_processed, load_raw
from preprocessing import preprocess_dataframe, get_default_feature_lists, build_preprocessor
from evaluate import plot_confusion_matrix, plot_roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "results" / "models"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 6, 12]
    }
    return clf, param_grid

def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    param_grid = {"clf__C": [0.01, 0.1, 1, 10]}
    return clf, param_grid

def main(args):
    # 1. Load Data
    # If raw path provided, force reload from raw
    if args.raw_path:
        print(f"Loading raw data from {args.raw_path}...")
        raw = load_raw(args.raw_path)
        X, y, preprocessor = preprocess_dataframe(raw)
        df = pd.concat([X, y.reset_index(drop=True)], axis=1)
        save_processed(df)
    else:
        try:
            df = load_processed()
            print("Loaded processed data.")
        except Exception:
            print("Processed file not found. Attempting to load default raw and preprocess.")
            raw = load_raw()
            X, y, preprocessor = preprocess_dataframe(raw)
            df = pd.concat([X, y.reset_index(drop=True)], axis=1)
            save_processed(df)

    # Reload to ensure consistency
    df = load_processed()
    X = df.drop(columns=["target"])
    y = df["target"]

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # 3. Build Pipeline
    # Derive feature lists from training data (prevents leakage)
    numeric_cols, cat_cols = get_default_feature_lists(pd.concat([X_train, y_train], axis=1))
    preprocessor = build_preprocessor(numeric_cols, cat_cols)

    if args.model == "rf":
        base_clf, param_grid = train_random_forest(X_train, y_train)
    else:
        base_clf, param_grid = train_logistic_regression(X_train, y_train)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", base_clf)
    ])

    # 4. Grid Search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=1)
    
    print(f"Starting GridSearchCV for {args.model.upper()}...")
    grid.fit(X_train, y_train)

    # 5. Show CV Results
    best = grid.best_estimator_
    print("\n" + "="*30)
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    
    if args.print_cv:
        print("-" * 30)
        print("Top 3 CV configurations:")
        results_df = pd.DataFrame(grid.cv_results_)
        results_df = results_df.sort_values(by="rank_test_score").head(3)
        for idx, row in results_df.iterrows():
            print(f"Rank {row['rank_test_score']}: {row['params']}")
            print(f"   Mean Score: {row['mean_test_score']:.4f} (Std: {row['std_test_score']:.4f})")
    print("="*30 + "\n")

    # 6. Evaluate on Test
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Test Set AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # 7. Save Artifacts
    model_path = MODELS_DIR / f"best_model_{args.model}.pkl"
    joblib.dump(best, model_path)
    print(f"Saved model to {model_path}")

    cm_fig = plot_confusion_matrix(y_test, y_pred, savepath=FIG_DIR / f"confusion_{args.model}.png")
    roc_fig = plot_roc_curve(y_test, y_proba, savepath=FIG_DIR / f"roc_{args.model}.png")
    print(f"Saved figures to {FIG_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "lr"], help="Which model to train: rf or lr")
    parser.add_argument("--raw-path", type=str, default=None, help="Path to raw CSV file (overrides auto-detection)")
    parser.add_argument("--print-cv", action="store_true", help="Print detailed cross-validation scores")
    args = parser.parse_args()
    main(args)