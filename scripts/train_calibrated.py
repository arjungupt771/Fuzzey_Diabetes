"""
Train a RandomForest and a calibrated classifier, and prepare inductive conformal calibration scores.
Saves:
 - model/model_rf_raw.pkl  (raw RandomForest)
 - model/model_calibrated.pkl (CalibratedClassifierCV)
 - model/conformal_calib.npy (nonconformity scores from calibration set)
 - model/feature_names.txt
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from diabetes_risk.ml_pipeline import make_target, select_features, make_pipeline
from diabetes_risk.reliability import fit_ood_detector


def main(data_path, out_dir='model', target_column=None):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    if target_column is None and 'Glucose' not in df.columns:
        raise SystemExit('Dataset must contain Glucose unless --target-column is specified')

    y, label_names, target_definition, target_column = make_target(df, target_column)
    X = select_features(df, target_column if target_definition.startswith('independent') else None)

    # split: proper training / calibration / test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    # so final: 60% train, 20% calib, 20% test

    rf = make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42))
    rf.fit(X_train, y_train)
    ood_detector = fit_ood_detector(X_train)

    # save raw model
    joblib.dump({
        'model': rf, 'feature_names': list(X.columns), 'label_names': label_names,
        'feature_defaults': X.median(numeric_only=True).fillna(0.0).to_dict(),
        'ood_detector': ood_detector,
        'target_definition': target_definition,
        'target_column': target_column,
        'target_feature_excluded': target_column,
        'fuzzy_uses_glucose': target_definition.startswith('independent'),
    }, os.path.join(out_dir, 'model_rf_raw.pkl'))

    # calibrated classifier using the proper training set; we'll use sigmoid
    base = make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42))
    try:
        calib = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=5)
    except TypeError:  # scikit-learn < 1.2
        calib = CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv=5)
    calib.fit(X_train, y_train)
    joblib.dump({'model': calib, 'feature_names': list(X.columns), 'label_names': label_names,
                 'feature_defaults': X.median(numeric_only=True).fillna(0.0).to_dict(),
                 'ood_detector': ood_detector,
                 'target_definition': target_definition, 'target_column': target_column,
                 'target_feature_excluded': target_column,
                 'fuzzy_uses_glucose': target_definition.startswith('independent')},
                os.path.join(out_dir, 'model_calibrated.pkl'))

    # compute inductive conformal nonconformity scores on calibration set using calibrated probabilities
    probs_calib = calib.predict_proba(X_calib)
    classes = list(calib.classes_)
    class_scores = {}
    for class_index, label in enumerate(classes):
        class_scores[int(label)] = (1.0 - probs_calib[y_calib.to_numpy() == label, class_index]).tolist()
    joblib.dump({'class_scores': class_scores, 'classes': classes}, os.path.join(out_dir, 'conformal_calib.pkl'))

    # save feature names
    feature_names = list(X.columns)
    with open(os.path.join(out_dir, 'feature_names.txt'), 'w', encoding='utf-8') as f:
        for c in feature_names:
            f.write(c + '\n')

    # save a small metadata file pointing to files
    meta = {
        'raw_model': 'model_rf_raw.pkl',
        'calibrated_model': 'model_calibrated.pkl',
        'conformal_calib': 'conformal_calib.pkl',
        'feature_names': feature_names,
        'feature_defaults': X.median(numeric_only=True).fillna(0.0).to_dict(),
        'classes': list(calib.classes_),
        'target_definition': target_definition,
        'target_column': target_column,
        'target_feature_excluded': target_column,
    }
    joblib.dump(meta, os.path.join(out_dir, 'model_metadata.pkl'))

    print(f'Target definition: {target_definition}')
    print('Saved raw model, calibrated model, class-conditional conformal scores, and metadata to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--out', type=str, default='model', help='Output directory')
    parser.add_argument('--target-column', type=str, default=None, help='Independent 3-class target column')
    args = parser.parse_args()
    main(args.data, args.out, args.target_column)
