"""Optional Optuna search for a Random Forest using stratified macro recall."""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from diabetes_risk.ml_pipeline import make_target, make_pipeline, select_features
from diabetes_risk.reliability import fit_ood_detector


def main(data_path, out_dir='model', trials=40, folds=5, target_column=None):
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit('Optuna is optional. Install it with `pip install optuna` to run this command.') from exc
    data = pd.read_csv(data_path)
    y, label_names, target_definition, target_column = make_target(data, target_column)
    X = select_features(data, target_column if target_definition.startswith('independent') else None)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scorer = make_scorer(recall_score, average='macro', zero_division=0)

    def objective(trial):
        estimator = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 600, step=100),
            max_depth=trial.suggest_categorical('max_depth', [None, 4, 6, 10, 16]),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 12),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 8),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            random_state=42, n_jobs=-1,
        )
        return cross_val_score(make_pipeline(estimator), X, y, cv=cv, scoring=scorer, n_jobs=1).mean()

    study = optuna.create_study(direction='maximize', study_name='rf_macro_recall')
    study.optimize(objective, n_trials=trials)
    best = RandomForestClassifier(**study.best_params, random_state=42, n_jobs=-1)
    model = make_pipeline(best).fit(X, y)
    ood_detector = fit_ood_detector(X)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({
        'model': model,
        'feature_names': list(X.columns),
        'feature_defaults': X.median(numeric_only=True).fillna(0.0).to_dict(),
        'ood_detector': ood_detector,
        'label_names': label_names,
        'selected_model': 'Random Forest (Optuna tuned)',
        'selection_metric': 'stratified CV macro recall',
        'best_cv_macro_recall': float(study.best_value),
        'best_parameters': study.best_params,
        'target_definition': target_definition,
        'target_column': target_column,
        'target_feature_excluded': target_column,
        'fuzzy_uses_glucose': target_definition.startswith('independent'),
    }, os.path.join(out_dir, 'model_rf_optuna.pkl'))
    study.trials_dataframe().to_csv(os.path.join(out_dir, 'optuna_trials.csv'), index=False)
    print(f'Best stratified CV macro recall: {study.best_value:.4f}')
    if not target_definition.startswith('independent'):
        print('Warning: tuning estimates use glucose-derived labels and are not screening performance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='model')
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--target-column', type=str, default=None, help='Independent 3-class target column')
    args = parser.parse_args()
    main(args.data, args.out, args.trials, args.folds, args.target_column)
