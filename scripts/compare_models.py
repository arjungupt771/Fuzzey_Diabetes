"""Compare available classifiers with stratified CV on leakage-safe features."""

import argparse
import os
import warnings

import joblib
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC

from diabetes_risk.ml_pipeline import make_target, make_pipeline, select_features
from diabetes_risk.reliability import fit_ood_detector


def candidate_models():
    candidates = {
        'Logistic Regression': (LogisticRegression(max_iter=2000, class_weight='balanced'), True),
        'Random Forest': (RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1), False),
        'SVM': (SVC(kernel='rbf', class_weight='balanced', probability=True), True),
    }
    optional = [
        ('XGBoost', 'xgboost', 'XGBClassifier'),
        ('LightGBM', 'lightgbm', 'LGBMClassifier'),
        ('CatBoost', 'catboost', 'CatBoostClassifier'),
    ]
    for title, module_name, class_name in optional:
        try:
            module = __import__(module_name, fromlist=[class_name])
            estimator = getattr(module, class_name)
            if title == 'XGBoost':
                model = estimator(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  objective='multi:softprob', eval_metric='mlogloss',
                                  random_state=42, n_jobs=-1)
            elif title == 'LightGBM':
                model = estimator(n_estimators=300, max_depth=-1, learning_rate=0.05,
                                  class_weight='balanced', random_state=42, verbosity=-1)
            else:
                model = estimator(iterations=300, depth=6, learning_rate=0.05,
                                  verbose=False, random_seed=42, auto_class_weights='Balanced')
            candidates[title] = (model, False)
        except ImportError:
            print(f'Skipping {title}: optional package {module_name} is not installed.')
    return candidates


def main(data_path, out_dir='model', n_splits=5, track_mlflow=False, target_column=None):
    df = pd.read_csv(data_path)
    y, label_names, target_definition, target_column = make_target(df, target_column)
    X = select_features(df, target_column if target_definition.startswith('independent') else None)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scorers = {
        'macro_recall': make_scorer(recall_score, average='macro', zero_division=0),
        'macro_f1': make_scorer(f1_score, average='macro', zero_division=0),
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
    }

    results = []
    fitted_candidates = {}
    for name, (estimator, scale) in candidate_models().items():
        pipeline = make_pipeline(estimator, scale=scale)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers,
                                    return_train_score=False, n_jobs=1, error_score='raise')
        row = {'model': name}
        for metric in scorers:
            values = scores[f'test_{metric}']
            row[f'{metric}_mean'] = float(values.mean())
            row[f'{metric}_std'] = float(values.std(ddof=1))
        results.append(row)
        fitted_candidates[name] = pipeline

    table = pd.DataFrame(results).sort_values(
        ['macro_recall_mean', 'macro_f1_mean', 'balanced_accuracy_mean'], ascending=False
    ).reset_index(drop=True)
    best_name = table.iloc[0]['model']
    selected_pipeline = fitted_candidates[best_name]
    best_model = clone(selected_pipeline).fit(X, y)
    ood_detector = fit_ood_detector(X)
    fold_importances = []
    for train_indices, test_indices in cv.split(X, y):
        fold_model = clone(selected_pipeline).fit(X.iloc[train_indices], y.iloc[train_indices])
        importance = permutation_importance(
            fold_model, X.iloc[test_indices], y.iloc[test_indices],
            scoring='f1_macro', n_repeats=5, random_state=42, n_jobs=1,
        )
        fold_importances.append(importance.importances_mean)
    importance_table = pd.DataFrame({
        'feature': X.columns,
        'permutation_f1_drop_mean': np.mean(fold_importances, axis=0),
        'permutation_f1_drop_std': np.std(fold_importances, axis=0, ddof=1),
    }).sort_values('permutation_f1_drop_mean', ascending=False)
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, 'model_comparison.csv'), index=False)
    importance_table.to_csv(os.path.join(out_dir, 'model_feature_importance.csv'), index=False)
    joblib.dump({
        'model': best_model,
        'feature_names': list(X.columns),
        'feature_defaults': X.median(numeric_only=True).fillna(0.0).to_dict(),
        'ood_detector': ood_detector,
        'label_names': label_names,
        'selected_model': best_name,
        'selection_metric': 'stratified CV macro recall, then macro F1, then balanced accuracy',
        'cv_results': table.to_dict(orient='records'),
        'target_definition': target_definition,
        'target_column': target_column,
        'target_feature_excluded': target_column,
        'fuzzy_uses_glucose': target_definition.startswith('independent'),
    }, os.path.join(out_dir, 'model_best_cv.pkl'))
    if track_mlflow:
        try:
            import mlflow
            import mlflow.sklearn
        except ImportError as exc:
            raise SystemExit('MLflow tracking is optional. Install it with `pip install mlflow`.') from exc
        with mlflow.start_run(run_name='diabetes_model_comparison'):
            mlflow.log_param('selected_model', best_name)
            mlflow.log_param('selection_metric', 'macro_recall_then_macro_f1')
            mlflow.log_param('cv_folds', n_splits)
            mlflow.log_param('target', target_definition)
            for row in results:
                prefix = row['model'].lower().replace(' ', '_')
                for metric, value in row.items():
                    if metric != 'model':
                        mlflow.log_metric(f'{prefix}_{metric}', value)
            mlflow.log_artifact(os.path.join(out_dir, 'model_comparison.csv'))
            mlflow.log_artifact(os.path.join(out_dir, 'model_feature_importance.csv'))
            mlflow.sklearn.log_model(best_model, artifact_path='selected_model')
    print(table.to_string(index=False))
    print(f'Best by macro recall then macro F1: {best_name}')
    if not target_definition.startswith('independent'):
        print('Warning: CV scores are model-selection estimates on glucose-derived labels, not screening performance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='CSV dataset with Glucose target values')
    parser.add_argument('--out', default='model', help='Directory for comparison artifacts')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--mlflow', action='store_true', help='Log the run and selected model to local/configured MLflow tracking.')
    parser.add_argument('--target-column', type=str, default=None, help='Independent 3-class target column')
    args = parser.parse_args()
    main(args.data, args.out, args.folds, args.mlflow, args.target_column)
