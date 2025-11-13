"""
Inductive Conformal Predictor helper for multiclass using saved calibration nonconformity scores.

Usage example:
    from conformal_predict import load_conformal, predict_set
    meta = load_conformal('model/model_metadata.pkl')
    preds = predict_set(meta, X_df, alpha=0.1)

This returns p-values and prediction sets for each row in X_df.
"""
import joblib
import numpy as np
import pandas as pd
import os


def load_conformal(meta_path='model/model_metadata.pkl'):
    m = joblib.load(meta_path)
    base = os.path.dirname(meta_path) or '.'
    calib = np.load(os.path.join(base, m['conformal_calib']))
    calib = np.asarray(calib)
    calibrated_model = joblib.load(os.path.join(base, m['calibrated_model']))
    feature_names = m.get('feature_names')
    return {'calib_nonconf': calib, 'model': calibrated_model, 'feature_names': feature_names}


def predict_pvalues(meta, X_df):
    # meta: dict returned by load_conformal
    model = meta['model']
    calib = meta['calib_nonconf']
    # ensure columns match feature names saved during training
    feature_names = meta.get('feature_names')
    if feature_names is not None:
        X_proc = X_df.copy()
        # select only known features and keep order
        X_proc = X_proc.reindex(columns=feature_names, fill_value=0.0)
    else:
        X_proc = X_df.copy()
    probs = model.predict_proba(X_proc)
    n_cal = len(calib)
    # p-values for each class candidate for each sample
    pvals = []
    for idx in range(probs.shape[0]):
        row_p = []
        for c in range(probs.shape[1]):
            nonconf = 1.0 - probs[idx, c]
            # p-value: fraction of calibration nonconformity >= nonconf (plus one for smoothing)
            p = (np.sum(calib >= nonconf) + 1) / (n_cal + 1)
            row_p.append(float(p))
        pvals.append(row_p)
    return np.array(pvals)


def prediction_sets_from_pvals(pvals, alpha=0.1):
    # pvals: n_samples x n_classes
    sets = []
    for row in pvals:
        labels = [int(i) for i, pv in enumerate(row) if pv > alpha]
        sets.append(labels)
    return sets


def predict_set(meta, X_df, alpha=0.1):
    pvals = predict_pvalues(meta, X_df)
    sets = prediction_sets_from_pvals(pvals, alpha=alpha)
    return {'pvalues': pvals, 'prediction_sets': sets}


if __name__ == '__main__':
    # quick smoke: load model and run on a sample row from dataset if present
    import pandas as pd
    if os.path.exists('..\\Fuzzy_Diabetes\\Model\\diabetes.csv'):
        df = pd.read_csv('..\\Fuzzy_Diabetes\\Model\\diabetes.csv')
        X = df.select_dtypes(include=[np.number]).fillna(df.median())
        X = X.head(5)
        meta = load_conformal('model/model_metadata.pkl')
        res = predict_set(meta, X, alpha=0.1)
        print('Prediction sets (alpha=0.1):')
        for i, s in enumerate(res['prediction_sets']):
            print(i, s)
