"""Held-out ablation of the prototype's model, fuzzy, calibration, and conformal parts."""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from diabetes_risk.ml_pipeline import make_target, make_pipeline, select_features
from diabetes_risk.utils import fuzzy_risk_assessment


def calibrated_classifier(estimator):
    try:
        return CalibratedClassifierCV(estimator=estimator, method='sigmoid', cv=5)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method='sigmoid', cv=5)


def labels_from_score(scores):
    # Illustrative thresholds only; the fuzzy system is not clinically calibrated.
    return np.where(scores < 0.34, 0, np.where(scores < 0.67, 1, 2))


def metrics_row(name, truth, predicted, **extra):
    return {
        'component': name,
        'accuracy': accuracy_score(truth, predicted),
        'macro_recall': recall_score(truth, predicted, labels=[0, 1, 2], average='macro', zero_division=0),
        'macro_f1': f1_score(truth, predicted, labels=[0, 1, 2], average='macro', zero_division=0),
        **extra,
    }


def main(data_path, out_path='Paper/ablation_results.csv', target_column=None):
    data = pd.read_csv(data_path)
    y, label_names, target_definition, target_column = make_target(data, target_column)
    label_values = {str(value).strip().lower().replace(' ', '-') for value in label_names.values()}
    if not {'normal', 'diabetic'}.issubset(label_values) or not label_values.intersection({'prediabetic', 'pre-diabetic'}):
        raise SystemExit('Fuzzy-to-class ablations require ordered normal/prediabetic/diabetic target labels.')
    X = select_features(data, target_column if target_definition.startswith('independent') else None)
    if 'BMI' not in data or 'Age' not in data:
        raise SystemExit('Ablation requires BMI and Age columns for the fuzzy component.')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    raw = make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)).fit(X_train, y_train)
    calibrated = calibrated_classifier(
        make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42))
    ).fit(X_train, y_train)
    raw_probabilities = raw.predict_proba(X_test)
    calibrated_probabilities = calibrated.predict_proba(X_test)
    truth = y_test.to_numpy()
    raw_predictions = raw.classes_[np.argmax(raw_probabilities, axis=1)]
    calibrated_predictions = calibrated.classes_[np.argmax(calibrated_probabilities, axis=1)]

    fuzzy_values = []
    for index in X_test.index:
        family_value = data.loc[index, 'FamilyHistory'] if 'FamilyHistory' in data else None
        if pd.isna(family_value):
            family_value = None
        elif isinstance(family_value, str):
            family_value = family_value.strip().lower() in {'1', 'yes', 'true', 'present'}
        fuzzy_values.append(fuzzy_risk_assessment(
            glucose=float(data.loc[index, 'Glucose']) if target_definition.startswith('independent') and 'Glucose' in data else None,
            bmi=float(data.loc[index, 'BMI']),
            age=float(data.loc[index, 'Age']),
            family_history=family_value,
        )['score'])
    fuzzy_scores = np.asarray(fuzzy_values)
    raw_fused = 0.6 * raw_probabilities[:, list(raw.classes_).index(2)] + \
                0.3 * raw_probabilities[:, list(raw.classes_).index(1)] + 0.1 * fuzzy_scores
    calibrated_fused = 0.6 * calibrated_probabilities[:, list(calibrated.classes_).index(2)] + \
                       0.3 * calibrated_probabilities[:, list(calibrated.classes_).index(1)] + 0.1 * fuzzy_scores

    class_scores = {}
    calib_probabilities = calibrated.predict_proba(X_calib)
    for class_index, label in enumerate(calibrated.classes_):
        class_scores[int(label)] = 1 - calib_probabilities[y_calib.to_numpy() == label, class_index]
    prediction_sets = []
    for row in calibrated_probabilities:
        selected = []
        for class_index, label in enumerate(calibrated.classes_):
            scores = class_scores[int(label)]
            pvalue = (np.count_nonzero(scores >= 1 - row[class_index]) + 1) / (len(scores) + 1)
            if pvalue > 0.1:
                selected.append(int(label))
        prediction_sets.append(selected)
    set_sizes = np.asarray([len(value) for value in prediction_sets])
    coverage = np.mean([int(label) in selected for label, selected in zip(truth, prediction_sets)])

    rows = [
        metrics_row('Random Forest only', truth, raw_predictions),
        metrics_row('Fuzzy only', truth, labels_from_score(fuzzy_scores)),
        metrics_row('Random Forest + fuzzy', truth, labels_from_score(raw_fused)),
        metrics_row('Random Forest + calibration', truth, calibrated_predictions),
        metrics_row('Random Forest + conformal', truth, calibrated_predictions,
                    conformal_coverage=coverage, mean_prediction_set_size=float(set_sizes.mean())),
        metrics_row('Complete hybrid (calibrated RF + fuzzy)', truth,
                    labels_from_score(calibrated_fused), conformal_coverage=coverage,
                    mean_prediction_set_size=float(set_sizes.mean())),
    ]
    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    result.to_csv(out_path, index=False)
    print(result.to_string(index=False))
    print('Fuzzy score class thresholds (<0.34, <0.67) are illustrative, not clinically calibrated.')
    print(f'Target: {target_definition}. Ablation results are not clinical validation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='Paper/ablation_results.csv')
    parser.add_argument('--target-column', default=None, help='Independent 3-class target column')
    args = parser.parse_args()
    main(args.data, args.out, args.target_column)
