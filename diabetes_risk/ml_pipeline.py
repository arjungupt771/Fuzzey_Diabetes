"""Shared target, feature-selection, and preprocessing helpers."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GLUCOSE_TARGET_FEATURE = "Glucose"
NON_FEATURE_COLUMNS = {"Outcome"}
LABELS = {0: "normal", 1: "prediabetic", 2: "diabetic"}


def make_labels_from_glucose(df, glucose_col=GLUCOSE_TARGET_FEATURE):
    """Create research labels from glucose thresholds (not independent diagnoses)."""
    glucose = pd.to_numeric(df[glucose_col], errors="coerce")
    if not np.isfinite(glucose.to_numpy(dtype=float)).all():
        raise ValueError(f"{glucose_col} contains missing, non-numeric, or non-finite values")
    return pd.Series(np.select(
        [glucose >= 126, glucose >= 100], [2, 1], default=0
    ), index=df.index, dtype=int)


def make_target(df, target_column=None, glucose_col=GLUCOSE_TARGET_FEATURE):
    """Return integer target labels, display names, and target provenance.

    Passing a target column uses its independently recorded three-class labels.
    """
    if target_column is None:
        return (make_labels_from_glucose(df, glucose_col), LABELS,
                f'{glucose_col} thresholds: <100 normal, 100-125 prediabetic, >=126 diabetic',
                glucose_col)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not in the dataset")
    values = df[target_column]
    if values.isna().any():
        raise ValueError(f"Independent target column '{target_column}' contains missing labels")
    unique_values = list(pd.unique(values))
    if len(unique_values) != 3:
        raise ValueError(f"Independent target column '{target_column}' must contain exactly 3 classes; found {len(unique_values)}")
    normalized = {str(value).strip().lower(): value for value in unique_values}
    canonical = {
        'normal': 0, 'non-diabetic': 0, 'nondiabetic': 0,
        'prediabetic': 1, 'pre-diabetic': 1, 'pre diabetic': 1,
        'prediabetes': 1, 'pre-diabetes': 1,
        'diabetic': 2, 'diabetes': 2,
    }
    if all(key in canonical for key in normalized):
        ordered_values = []
        for category in ('normal', 'prediabetic', 'diabetic'):
            matches = [value for name, value in normalized.items() if canonical[name] == canonical[category]]
            if matches:
                ordered_values.extend(matches)
        if len(ordered_values) == 3:
            class_names = {index: str(value) for index, value in enumerate(ordered_values)}
            value_to_index = {value: index for index, value in enumerate(ordered_values)}
        else:
            ordered_values = sorted(unique_values, key=str)
            value_to_index = {value: index for index, value in enumerate(ordered_values)}
            class_names = {index: str(value) for index, value in enumerate(ordered_values)}
    elif set(unique_values) == {0, 1, 2}:
        value_to_index = {0: 0, 1: 1, 2: 2}
        class_names = LABELS.copy()
    else:
        ordered_values = sorted(unique_values, key=str)
        value_to_index = {value: index for index, value in enumerate(ordered_values)}
        class_names = {index: str(value) for index, value in enumerate(ordered_values)}
    encoded = values.map(value_to_index).astype(int)
    return encoded, class_names, f'independent labels from column {target_column}', target_column


def select_features(df, target_column=None, glucose_col=GLUCOSE_TARGET_FEATURE):
    """Select numeric predictors and exclude target-defining/leakage columns."""
    excluded = NON_FEATURE_COLUMNS | ({glucose_col} if target_column is None else {target_column})
    X = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in excluded if c in df.columns], errors="ignore"
    ).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how='all')
    if X.empty:
        raise ValueError("No numeric predictor features remain after excluding target columns")
    return X


def make_preprocessor():
    """Impute missing numeric values using statistics learned on each train fold."""
    return SimpleImputer(strategy="median")


def make_pipeline(estimator, scale=False):
    steps = [("imputer", make_preprocessor())]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)
