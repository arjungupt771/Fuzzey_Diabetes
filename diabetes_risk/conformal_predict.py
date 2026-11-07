"""Inductive conformal prediction helpers with class-conditional calibration."""

import os

import joblib
import numpy as np


def load_conformal(meta_path='model/model_metadata.pkl'):
    metadata = joblib.load(meta_path)
    base = os.path.dirname(meta_path) or '.'
    calibration_path = os.path.join(base, metadata['conformal_calib'])
    if calibration_path.endswith('.pkl'):
        calibration = joblib.load(calibration_path)
    else:  # Read earlier pooled-score artifacts for backward compatibility.
        calibration = {'scores': np.asarray(np.load(calibration_path)), 'class_scores': None}
    saved_model = joblib.load(os.path.join(base, metadata['calibrated_model']))
    model = saved_model.get('model') if isinstance(saved_model, dict) else saved_model
    return {
        'class_scores': calibration.get('class_scores'),
        'calib_nonconf': calibration.get('scores'),
        'model': model,
        'feature_names': metadata.get('feature_names'),
        'classes': list(model.classes_),
    }


def predict_pvalues(meta, X_df):
    feature_names = meta.get('feature_names')
    if feature_names is not None:
        missing = [name for name in feature_names if name not in X_df.columns]
        if missing:
            raise ValueError(f"Missing required model features: {', '.join(missing)}")
        X_proc = X_df.reindex(columns=feature_names)
    else:
        X_proc = X_df
    probabilities = meta['model'].predict_proba(X_proc)
    class_scores = meta.get('class_scores')
    pooled_scores = meta.get('calib_nonconf')
    pvalues = np.zeros_like(probabilities, dtype=float)
    for row_index, row in enumerate(probabilities):
        for class_index, label in enumerate(meta['classes']):
            scores = (np.asarray(class_scores.get(int(label), []))
                      if class_scores is not None else np.asarray(pooled_scores))
            if scores.size:
                candidate_nonconformity = 1.0 - row[class_index]
                pvalues[row_index, class_index] = (
                    np.count_nonzero(scores >= candidate_nonconformity) + 1
                ) / (len(scores) + 1)
    return pvalues


def prediction_sets_from_pvals(pvals, alpha=0.1, classes=None):
    classes = list(range(pvals.shape[1])) if classes is None else list(classes)
    return [[int(classes[i]) for i, value in enumerate(row) if value > alpha] for row in pvals]


def predict_set(meta, X_df, alpha=0.1):
    pvalues = predict_pvalues(meta, X_df)
    return {
        'pvalues': pvalues,
        'prediction_sets': prediction_sets_from_pvals(pvalues, alpha, meta.get('classes')),
    }
