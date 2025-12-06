"""
src/features.py
Simple feature helpers (selection / importances)
"""
import pandas as pd
import numpy as np

def correlation_with_target(df: pd.DataFrame, target_col: str = "target"):
    """
    Return absolute correlations of numeric features with target (descending).
    """
    if target_col not in df.columns:
        raise KeyError("target column missing")
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()[target_col].abs().sort_values(ascending=False)
    return corr.drop(labels=[target_col], errors="ignore")

def select_top_k_by_corr(df: pd.DataFrame, k: int = 8, target_col: str = "target"):
    corr = correlation_with_target(df, target_col)
    top = corr.iloc[:k].index.tolist()
    return top

def feature_names_after_preprocessing(preprocessor, numeric_columns, categorical_columns):
    """
    Reconstruct feature names after ColumnTransformer with OneHotEncoder (approximate).
    This helps map feature importances back to original names.
    """
    feature_names = []
    # numeric
    feature_names.extend(numeric_columns)
    # categorical: if onehot present, expand
    # try to access transformers
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            # trans could be a Pipeline
            ohe = None
            # search for OneHotEncoder inside pipeline
            if hasattr(trans, 'named_steps'):
                for step in trans.named_steps.values():
                    if type(step).__name__ == "OneHotEncoder":
                        ohe = step
                        break
            if ohe is not None:
                # get categories
                categories = ohe.categories_
                for col, cats in zip(cols, categories):
                    feature_names.extend([f"{col}__{str(val)}" for val in cats])
            else:
                # fallback: just add categorical col names
                feature_names.extend(cols)
    return feature_names
