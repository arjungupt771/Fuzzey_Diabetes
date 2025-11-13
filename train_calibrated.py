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
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


def make_labels_from_glucose(df, glucose_col='Glucose'):
    g = df[glucose_col].astype(float)
    labels = pd.Series(0, index=df.index)
    labels[(g >= 100) & (g < 126)] = 1
    labels[g >= 126] = 2
    return labels


def main(data_path, out_dir='model'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    if 'Glucose' not in df.columns:
        raise SystemExit('Dataset must contain Glucose column')

    y = make_labels_from_glucose(df, 'Glucose')
    X = df.select_dtypes(include=[np.number]).copy()
    if 'Outcome' in X.columns:
        X = X.drop(columns=['Outcome'])
    X = X.fillna(X.median())

    # split: proper training / calibration / test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    # so final: 60% train, 20% calib, 20% test

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # save raw model
    joblib.dump(rf, os.path.join(out_dir, 'model_rf_raw.pkl'))

    # calibrated classifier using the proper training set; we'll use sigmoid
    calib = CalibratedClassifierCV(RandomForestClassifier(n_estimators=200, random_state=42), method='sigmoid', cv=5)
    calib.fit(X_train, y_train)
    joblib.dump(calib, os.path.join(out_dir, 'model_calibrated.pkl'))

    # compute inductive conformal nonconformity scores on calibration set using calibrated probabilities
    probs_calib = calib.predict_proba(X_calib)
    # nonconformity = 1 - probability assigned to true class
    nonconformity = 1.0 - np.array([probs_calib[i, y_calib.iloc[i]] for i in range(len(y_calib))])
    np.save(os.path.join(out_dir, 'conformal_calib.npy'), nonconformity)

    # save feature names
    feature_names = list(X.columns)
    with open(os.path.join(out_dir, 'feature_names.txt'), 'w', encoding='utf-8') as f:
        for c in feature_names:
            f.write(c + '\n')

    # save a small metadata file pointing to files
    meta = {
        'raw_model': 'model_rf_raw.pkl',
        'calibrated_model': 'model_calibrated.pkl',
        'conformal_calib': 'conformal_calib.npy',
        'feature_names': feature_names
    }
    joblib.dump(meta, os.path.join(out_dir, 'model_metadata.pkl'))

    print('Saved raw model, calibrated model, conformal calibration scores, and metadata to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV with Glucose')
    parser.add_argument('--out', type=str, default='model', help='Output directory')
    args = parser.parse_args()
    main(args.data, args.out)
