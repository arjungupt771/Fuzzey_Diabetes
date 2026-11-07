"""Evaluate a saved model on a separate cohort without refitting or imputation leakage."""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score)

from diabetes_risk.ml_pipeline import LABELS, make_labels_from_glucose, select_features
from diabetes_risk.predict import load_model
from diabetes_risk.reliability import assess_ood
from diabetes_risk.conformal_predict import load_conformal, predict_set


def encode_external_target(series, label_names):
    reverse = {str(name).strip().lower(): int(code) for code, name in label_names.items()}
    if set(series.dropna().unique()) <= {0, 1, 2}:
        mapping = {value: int(value) for value in series.dropna().unique()}
    else:
        mapping = {}
        for value in series.dropna().unique():
            key = str(value).strip().lower()
            if key not in reverse:
                raise ValueError(f"External target value '{value}' is not in the model's label mapping")
            mapping[value] = reverse[key]
    if series.isna().any():
        raise ValueError('External target contains missing labels')
    return series.map(mapping).astype(int)


def main(data_path, model_path='model/model_best_cv.pkl', target_column=None, output='Paper/external_validation.md'):
    artifact = joblib.load(model_path)
    model, feature_names = load_model(model_path)
    if not isinstance(artifact, dict):
        raise ValueError('Model artifact must include feature and target metadata; retrain with the current pipeline.')
    label_names = artifact.get('label_names', LABELS)
    target_definition = artifact.get('target_definition', '')
    stored_target = artifact.get('target_column')
    external = pd.read_csv(data_path)
    target_column = target_column or stored_target
    if target_definition.startswith('Glucose thresholds'):
        y = make_labels_from_glucose(external)
        feature_target = None
    else:
        if not target_column or target_column not in external.columns:
            raise ValueError('Supply the independent target column used to train the saved model.')
        y = encode_external_target(external[target_column], label_names)
        feature_target = target_column
    X = select_features(external, feature_target)
    missing = [name for name in feature_names if name not in X.columns]
    if missing:
        raise ValueError(f'External cohort is missing required predictors: {", ".join(missing)}')
    X = X.reindex(columns=feature_names)
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    labels = sorted(set(classes) | set(y.unique()))
    metrics = {
        'n': int(len(y)),
        'target_definition': target_definition,
        'accuracy': float(accuracy_score(y, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'macro_precision': float(precision_score(y, predicted, labels=labels, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y, predicted, labels=labels, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y, predicted, labels=labels, average='macro', zero_division=0)),
        'class_distribution': {label_names.get(int(code), str(code)): int((y == code).sum()) for code in labels},
    }
    detector = artifact.get('ood_detector')
    if detector is not None:
        checks = [assess_ood(detector, X.iloc[[idx]]) for idx in range(len(X))]
        metrics['unusual_input_rate'] = float(np.mean([item['is_unusual'] for item in checks]))
    metadata_path = os.path.join(os.path.dirname(model_path) or '.', 'model_metadata.pkl')
    if os.path.basename(model_path) == 'model_calibrated.pkl' and os.path.exists(metadata_path):
        conformal = load_conformal(metadata_path)
        conformal_result = predict_set(conformal, X, alpha=0.1)
        sets = conformal_result['prediction_sets']
        metrics['conformal_nominal_coverage'] = 0.9
        metrics['conformal_external_coverage'] = float(np.mean([
            int(truth) in selected for truth, selected in zip(y.to_numpy(), sets)
        ]))
        metrics['conformal_mean_set_size'] = float(np.mean([len(selected) for selected in sets]))
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w', encoding='utf-8') as handle:
        handle.write('# External cohort evaluation\n\n')
        handle.write(f"Cohort: `{os.path.abspath(data_path)}`. Model: `{os.path.abspath(model_path)}`.\n\n")
        handle.write('This report evaluates a fixed saved model without refitting on the external cohort. Confirm label provenance and class mapping before interpreting metrics.\n\n')
        handle.write('```json\n' + json.dumps(metrics, indent=2) + '\n```\n\n')
        handle.write('## Classification report\n\n```text\n')
        handle.write(classification_report(y, predicted, labels=labels,
                                           target_names=[label_names.get(int(c), str(c)) for c in labels],
                                           zero_division=0))
        handle.write('\n```\n')
    print(f'Wrote external validation report to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='CSV from an independent cohort')
    parser.add_argument('--model', default='model/model_best_cv.pkl')
    parser.add_argument('--target-column', default=None, help='Independent target column, if used in training')
    parser.add_argument('--out', default='Paper/external_validation.md')
    args = parser.parse_args()
    main(args.data, args.model, args.target_column, args.out)
