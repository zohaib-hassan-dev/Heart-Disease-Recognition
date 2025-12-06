"""
src/preprocessing.py
Preprocessing utilities: imputation, encoding, scaling and a reusable column transformer.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

def get_default_feature_lists(df: pd.DataFrame):
    """
    Return lists of numeric and categorical columns (best-effort).
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # treat 'target' as not a feature
    numeric_cols = [c for c in numeric_cols if c != "target"]
    # categorical are the rest
    cat_cols = [c for c in df.columns if c not in numeric_cols + ["target"]]
    return numeric_cols, cat_cols

def build_preprocessor(numeric_columns, categorical_columns, scale_numeric=True, onehot=True):
    """
    Build a ColumnTransformer with simple imputation and encoding/scaling.
    Returns the transformer object.
    """
    # numeric pipeline
    numeric_transformers = []
    numeric_transformers.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        numeric_transformers.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_transformers)

    # categorical pipeline
    # use SimpleImputer + OneHotEncoder (handle_unknown='ignore')
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    if onehot:
        # scikit-learn changed the OneHotEncoder API: older versions accept `sparse`,
        # newer ones use `sparse_output`. Try both to remain compatible.
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_pipeline.steps.append(("ohe", ohe))
    else:
        cat_pipeline.steps.append(("ord", OrdinalEncoder()))
    transformer = ColumnTransformer([
        ("num", numeric_pipeline, numeric_columns),
        ("cat", cat_pipeline, categorical_columns)
    ], remainder="drop", sparse_threshold=0)  # ensure output is ndarray
    return transformer

def preprocess_dataframe(df: pd.DataFrame):
    """
    Minimal cleaning & returns X, y and the preprocessor configured for those columns.
    This function assumes the label column is named 'target'.
    """
    df = df.copy()

    # Accept common label column names: if `target` missing, try `num` (UCI dataset)
    if "target" not in df.columns:
        if "num" in df.columns:
            df = df.rename(columns={"num": "target"})
            print("Info: renamed 'num' column to 'target' for compatibility.")
        else:
            # try to auto-detect a binary label column (0/1 or '0'/'1')
            detected = None
            for col in df.columns:
                vals = pd.Series(df[col].dropna().unique())
                try:
                    vals_set = set(vals.astype(str).tolist())
                except Exception:
                    vals_set = set(map(str, vals.tolist()))
                if vals_set.issubset({"0", "1"}):
                    detected = col
                    break
            if detected:
                df = df.rename(columns={detected: "target"})
                print(f"Info: auto-detected label column '{detected}' and renamed to 'target'.")

    if "target" not in df.columns:
        raise KeyError("Expected 'target' column in dataframe.")

    # drop obvious identifier columns
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # ensure target is binary 0/1. Some UCI variants use 0..4 to indicate severity.
    try:
        df["target"] = pd.to_numeric(df["target"]).astype(int)
        uniques = sorted(df["target"].dropna().unique().tolist())
        if any(u not in (0, 1) for u in uniques):
            # map >0 to 1
            df["target"] = df["target"].apply(lambda v: 1 if int(v) > 0 else 0)
            print("Info: converted multi-class 'target' to binary (0/1).")
    except Exception:
        # leave as-is; downstream will fail if it's not usable
        pass

    # basic cleaning: strip column names
    df.columns = [c.strip() for c in df.columns]

    # handle obvious dtype issues (try converting numeric columns)
    for col in df.columns:
        if df[col].dtype == object:
            # try to coerce to numeric where possible
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    # split
    X = df.drop(columns=["target"])
    y = df["target"]
    numeric_cols, cat_cols = get_default_feature_lists(df)
    # ensure cat columns don't include numeric by mistake
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    return X, y, preprocessor
