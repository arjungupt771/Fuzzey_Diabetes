"""
Train a RandomForest classifier for 3 classes: normal / prediabetic / diabetic.
Label mapping is created from the 'Glucose' column using ADA-like thresholds.
Saves trained model to given output path.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def make_labels_from_glucose(df, glucose_col='Glucose'):
    """Create 3-class labels from glucose values:
    0 -> normal (glucose < 100)
    1 -> prediabetic (100 <= glucose < 126)
    2 -> diabetic (glucose >= 126)
    """
    g = df[glucose_col].astype(float)
    labels = pd.Series(0, index=df.index)
    labels[(g >= 100) & (g < 126)] = 1
    labels[g >= 126] = 2
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Path to CSV dataset (must contain Glucose column)')
    parser.add_argument('--out', type=str, default='model/model_rf.pkl', help='Output model path')
    args = parser.parse_args()

    if args.data is None:
        raise SystemExit('Please provide --data pointing to a CSV dataset with a Glucose column')

    df = pd.read_csv(args.data)
    if 'Glucose' not in df.columns:
        raise SystemExit('Dataset must contain a Glucose column for label generation')

    y = make_labels_from_glucose(df, 'Glucose')

    # Drop non-numeric or outcome columns if present
    X = df.select_dtypes(include=[np.number]).copy()
    # Remove original Outcome column if exists to avoid label leakage
    if 'Outcome' in X.columns:
        X = X.drop(columns=['Outcome'])

    # Fill NA with median
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, digits=4))

    joblib.dump({'model': clf, 'feature_names': list(X.columns)}, args.out)
    print(f'Model saved to {args.out}')


if __name__ == '__main__':
    main()
