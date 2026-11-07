"""Evaluate raw and calibrated RF variants on a deterministic held-out split."""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from diabetes_risk.ml_pipeline import LABELS, make_target, make_pipeline, select_features


def multiclass_brier(y_true, probabilities, classes):
    onehot = np.zeros((len(y_true), len(classes)))
    class_indices = {int(label): i for i, label in enumerate(classes)}
    for row, label in enumerate(y_true):
        onehot[row, class_indices[int(label)]] = 1
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, probabilities, classes, bins=10):
    errors = []
    for column, label in enumerate(classes):
        confidence = probabilities[:, column]
        correct = (np.asarray(y_true) == label).astype(float)
        edges = np.linspace(0, 1, bins + 1)
        class_error = 0.0
        for i in range(bins):
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < bins - 1 else confidence <= edges[i + 1])
            if mask.any():
                class_error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
        errors.append(class_error)
    return float(np.mean(errors))


def calibrated_model(estimator, method):
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=5)
    except TypeError:  # scikit-learn < 1.2
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=5)


def bootstrap_macro_metrics(y_true, y_pred, repeats=1000, seed=42):
    rng = np.random.default_rng(seed)
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    recalls, f1s = [], []
    for _ in range(repeats):
        indices = rng.integers(0, len(truth), len(truth))
        recalls.append(recall_score(truth[indices], prediction[indices], labels=list(LABELS),
                                    average='macro', zero_division=0))
        f1s.append(f1_score(truth[indices], prediction[indices], labels=list(LABELS),
                            average='macro', zero_division=0))
    return np.percentile(recalls, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5])


def main(data_path, out_dir='Paper', target_column=None):
    data = pd.read_csv(data_path)
    y, label_names, target_definition, target_column = make_target(data, target_column)
    X = select_features(data, target_column if target_definition.startswith('independent') else None)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    raw = make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42))
    candidates = {'raw': raw}
    for method in ('sigmoid', 'isotonic'):
        candidates[method] = calibrated_model(
            make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)), method
        )

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'figures'), exist_ok=True)
    result_rows = []
    predictions = {}
    label_predictions = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)
        classes = list(model.classes_)
        predictions[name] = (probabilities, classes)
        y_pred = np.asarray(classes)[np.argmax(probabilities, axis=1)]
        label_predictions[name] = y_pred
        recall_interval, f1_interval = bootstrap_macro_metrics(y_test.to_numpy(), y_pred)
        result_rows.append({
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_recall': recall_score(y_test, y_pred, labels=list(LABELS), average='macro', zero_division=0),
            'macro_recall_95ci_low': recall_interval[0],
            'macro_recall_95ci_high': recall_interval[1],
            'macro_f1': f1_score(y_test, y_pred, labels=list(LABELS), average='macro', zero_division=0),
            'macro_f1_95ci_low': f1_interval[0],
            'macro_f1_95ci_high': f1_interval[1],
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'multiclass_brier': multiclass_brier(y_test.to_numpy(), probabilities, classes),
            'macro_ece': expected_calibration_error(y_test.to_numpy(), probabilities, classes),
        })

    metrics = pd.DataFrame(result_rows).set_index('model')
    metrics.to_csv(os.path.join(out_dir, 'results_table.csv'))
    paired_rng = np.random.default_rng(2026)
    paired_differences = []
    truth_array = y_test.to_numpy()
    for _ in range(1000):
        sample = paired_rng.integers(0, len(truth_array), len(truth_array))
        sig_f1 = f1_score(truth_array[sample], label_predictions['sigmoid'][sample],
                          labels=list(LABELS), average='macro', zero_division=0)
        raw_f1 = f1_score(truth_array[sample], label_predictions['raw'][sample],
                          labels=list(LABELS), average='macro', zero_division=0)
        paired_differences.append(sig_f1 - raw_f1)
    paired_f1_difference_ci = np.percentile(paired_differences, [2.5, 97.5])

    subgroup_rows = []
    for feature, bins in [('Age', [0, 35, 50, np.inf]), ('BMI', [0, 25, 30, np.inf])]:
        if feature not in X_test.columns:
            continue
        groups = pd.cut(X_test[feature], bins=bins, right=False, include_lowest=True)
        for group in groups.dropna().unique():
            mask = (groups == group).to_numpy()
            if mask.sum() < 5:
                continue
            subgroup_rows.append({
                'feature': feature,
                'group': str(group),
                'n': int(mask.sum()),
                'sigmoid_macro_recall': recall_score(
                    truth_array[mask], label_predictions['sigmoid'][mask], labels=list(LABELS),
                    average='macro', zero_division=0,
                ),
                'sigmoid_macro_f1': f1_score(
                    truth_array[mask], label_predictions['sigmoid'][mask], labels=list(LABELS),
                    average='macro', zero_division=0,
                ),
            })
    pd.DataFrame(subgroup_rows, columns=['feature', 'group', 'n', 'sigmoid_macro_recall', 'sigmoid_macro_f1']).to_csv(
        os.path.join(out_dir, 'subgroup_metrics.csv'), index=False
    )
    for column, class_label in enumerate(predictions['raw'][1]):
        plt.figure(figsize=(6, 4))
        for name, (probabilities, classes) in predictions.items():
            class_column = classes.index(class_label)
            fraction_positive, mean_predicted = calibration_curve(
                (y_test.to_numpy() == class_label).astype(int), probabilities[:, class_column],
                n_bins=10, strategy='uniform'
            )
            plt.plot(mean_predicted, fraction_positive, marker='o', label=name)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Observed class frequency')
        plt.title(f'Calibration: {label_names[int(class_label)]}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'figures', f'calibration_class_{class_label}.png'))
        plt.close()

    conformal_summary = None
    metadata_path = os.path.join('model', 'model_metadata.pkl')
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        calibration_file = os.path.join('model', metadata['conformal_calib'])
        if calibration_file.endswith('.pkl') and metadata.get('target_definition') == target_definition:
            calibration = joblib.load(calibration_file)
            class_scores = calibration.get('class_scores')
            calibrated = joblib.load(os.path.join('model', metadata['calibrated_model']))
            calibrated = calibrated.get('model') if isinstance(calibrated, dict) else calibrated
            test_probs = calibrated.predict_proba(X_test.reindex(columns=metadata['feature_names']))
            class_values = list(calibrated.classes_)
            sets = []
            for row in test_probs:
                selected = []
                for class_index, label in enumerate(class_values):
                    scores = np.asarray(class_scores.get(int(label), []))
                    if scores.size:
                        pvalue = (np.count_nonzero(scores >= 1.0 - row[class_index]) + 1) / (len(scores) + 1)
                        if pvalue > 0.1:
                            selected.append(int(label))
                sets.append(selected)
            set_sizes = np.asarray([len(value) for value in sets])
            covered = np.asarray([int(truth) in prediction_set for truth, prediction_set in zip(y_test, sets)])
            conformal_summary = {
                'nominal_coverage': 0.9,
                'observed_test_coverage': float(covered.mean()),
                'mean_set_size': float(set_sizes.mean()),
                'empty_set_rate': float(np.mean(set_sizes == 0)),
                'singleton_rate': float(np.mean(set_sizes == 1)),
            }

    with open(os.path.join(out_dir, 'results_summary.md'), 'w', encoding='utf-8') as handle:
        handle.write('# Held-out results summary\n\n')
        handle.write(f'All models were evaluated on the same deterministic stratified 20% holdout; imputation is learned from training folds. Target: {target_definition}. '
                     'Results do not establish clinical utility or generalizability.\n\n')
        handle.write('```text\n' + metrics.to_string() + '\n```')
        handle.write('\n\nPaired bootstrap 95% interval for sigmoid minus raw macro-F1: '
                     f'[{paired_f1_difference_ci[0]:.4f}, {paired_f1_difference_ci[1]:.4f}]. '
                     'This interval is exploratory and does not establish clinical superiority.\n')
        handle.write('\nSubgroup metrics are reported only for held-out age/BMI groups with at least five rows. '
                     'Small exploratory groups are not fairness validation. See `subgroup_metrics.csv`.\n')
        handle.write('\n\n## Conformal prediction\n\n')
        if conformal_summary:
            for key, value in conformal_summary.items():
                handle.write(f'- {key}: {value:.4f}\n')
        else:
            handle.write('No current class-conditional conformal artifacts were found.\n')
        handle.write('\n## Calibration plots\n\n')
        for class_label in predictions['raw'][1]:
            handle.write(f'### {label_names[int(class_label)]}\n\n')
            handle.write(f'![Calibration plot](figures/calibration_class_{class_label}.png)\n\n')
        handle.write('Probability calibration and conformal coverage are dataset-level statistical summaries, not clinical confidence guarantees.\n')
    print(f'Wrote held-out results under {out_dir}')
    print('Target labels are glucose-threshold-derived, not independently diagnosed outcomes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='..\\Fuzzy_Diabetes\\Model\\diabetes.csv')
    parser.add_argument('--out', default='Paper')
    parser.add_argument('--target-column', default=None, help='Independent 3-class target column')
    args = parser.parse_args()
    main(args.data, args.out, args.target_column)
