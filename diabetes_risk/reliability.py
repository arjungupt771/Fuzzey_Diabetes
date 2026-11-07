"""Lightweight input-distribution checks for the research prototype."""

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_ood_detector(X, contamination=0.05, random_state=42):
    """Fit an Isolation Forest on training predictors only."""
    detector = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('isolation_forest', IsolationForest(
            contamination=contamination, random_state=random_state
        )),
    ])
    return detector.fit(X)


def assess_ood(detector, X):
    if detector is None:
        return None
    prediction = int(detector.predict(X)[0])
    return {
        'is_unusual': prediction == -1,
        'score': float(detector.decision_function(X)[0]),
    }
