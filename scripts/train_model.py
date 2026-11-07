"""
Train a RandomForest classifier for 3 classes: normal / prediabetic / diabetic.
Label mapping is created from the 'Glucose' column using ADA-like thresholds.
Saves trained model to given output path.
"""

import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from diabetes_risk.ml_pipeline import make_target, select_features, make_pipeline
from diabetes_risk.reliability import fit_ood_detector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='Path to CSV dataset')
    parser.add_argument('--out', type=str, default='model/model_rf.pkl', help='Output model path')
    parser.add_argument('--target-column', type=str, default=None, help='Independent 3-class target column; omit to derive research labels from Glucose')
    args = parser.parse_args()

    if args.data is None:
        raise SystemExit('Please provide --data pointing to a CSV dataset')

    df = pd.read_csv(args.data)
    if args.target_column is None and 'Glucose' not in df.columns:
        raise SystemExit('Dataset must contain Glucose unless --target-column is specified')

    y, label_names, target_definition, target_column = make_target(df, args.target_column)
    X = select_features(df, args.target_column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42))
    clf.fit(X_train, y_train)
    ood_detector = fit_ood_detector(X_train)

    preds = clf.predict(X_test)
    if args.target_column is None:
        print('Target labels are glucose-threshold-derived; this is not independent diagnostic validation.')
    class_order = sorted(label_names)
    print(classification_report(y_test, preds, labels=class_order,
                                target_names=[label_names[label] for label in class_order],
                                digits=4, zero_division=0))

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump({
        'model': clf,
        'feature_names': list(X.columns),
        'feature_defaults': X.median(numeric_only=True).fillna(0.0).to_dict(),
        'ood_detector': ood_detector,
        'label_names': label_names,
        'target_definition': target_definition,
        'target_column': target_column,
        'target_feature_excluded': target_column,
        'fuzzy_uses_glucose': args.target_column is not None,
    }, args.out)
    print(f'Model saved to {args.out}')


if __name__ == '__main__':
    main()
