from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from config import (
    LEAKAGE_COLS,
    CAT_COLS,
    LABEL_COL,
    ATTACK_CAT_COL,
)


def load_raw(path) -> pd.DataFrame:
    """Read Parquet/CSV and normalize column names."""
    path = Path(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    df.columns = df.columns.astype(str).str.strip()

    # Normalize categorical values; preserve missing values for imputation.
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().fillna("__MISSING__")

    return df


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leakage/identifier columns only when present."""
    return df.drop(
        columns=[col for col in LEAKAGE_COLS if col in df.columns],
        errors="ignore",
    )


def split_features_target(df: pd.DataFrame):
    """Return X, y and attack category metadata without target leakage."""
    if LABEL_COL not in df.columns:
        raise ValueError(f"Required target column {LABEL_COL!r} is missing.")

    # attack_cat is metadata for per-category reporting, never a feature.
    y = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int)
    unexpected = set(y.unique()) - {0, 1}
    if unexpected:
        raise ValueError(f"Expected binary labels 0/1; found {sorted(unexpected)}")

    attack_cat = (
        df[ATTACK_CAT_COL].astype("string").str.strip().fillna("Unknown")
        if ATTACK_CAT_COL in df.columns
        else pd.Series(["Unknown"] * len(df), index=df.index, dtype="string")
    )

    excluded = {LABEL_COL, ATTACK_CAT_COL}
    X = df.drop(columns=[c for c in excluded if c in df.columns]).copy()

    # Ensure known categorical columns exist and are consistently represented.
    categorical_cols = [c for c in CAT_COLS if c in X.columns]
    for col in categorical_cols:
        X[col] = X[col].astype("string").str.strip().fillna("__MISSING__")

    # Convert remaining features to numeric, and convert +/-inf to NaN.
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    return X, y, attack_cat, categorical_cols, numeric_cols


def build_preprocessor(categorical_cols, numeric_cols):
    """Create a train-fitted transformer; unseen test categories are ignored."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessors(train_df: pd.DataFrame):
    """Fit preprocessing using training data only."""
    X_train, y_train, _, cat_cols, num_cols = split_features_target(train_df)
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out().tolist()

    artifacts = {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "categorical_cols": cat_cols,
        "numeric_cols": num_cols,
        "input_feature_names": X_train.columns.tolist(),
    }
    return artifacts, X_train_transformed, y_train.to_numpy()


def transform(df: pd.DataFrame, artifacts):
    """Transform a dataset with the already-fitted training preprocessor."""
    X, y, attack_cat, _, _ = split_features_target(df)

    expected = artifacts["input_feature_names"]
    missing = [c for c in expected if c not in X.columns]
    extra = [c for c in X.columns if c not in expected]
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {missing}")
    if extra:
        raise ValueError(f"Dataset has unexpected feature columns: {extra}")

    # Match training column order exactly.
    X = X[expected]
    Xt = artifacts["preprocessor"].transform(X)
    return Xt, y.to_numpy(), attack_cat


def save_artifacts(artifacts, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, out_dir / "preprocessing.pkl")
    joblib.dump(artifacts["feature_names"], out_dir / "feature_names.pkl")


def load_artifacts(model_dir):
    model_dir = Path(model_dir)
    artifacts = joblib.load(model_dir / "preprocessing.pkl")
    return artifacts
