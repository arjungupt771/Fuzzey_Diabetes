"""Generate a results summary comparing raw RF, original RF, and calibrated classifier.
Saves metrics and calibration plots to Paper/ and Paper/figures.
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.calibration import calibration_curve


def make_labels_from_glucose(df, glucose_col='Glucose'):
    g = df[glucose_col].astype(float)
    labels = pd.Series(0, index=df.index)
    labels[(g >= 100) & (g < 126)] = 1
    labels[g >= 126] = 2
    return labels


def multiclass_brier_score(y_true, probs):
    # y_true: array-like of shape (n_samples,), probs: (n_samples, n_classes)
    n_classes = probs.shape[1]
    onehot = np.zeros_like(probs)
    for i, t in enumerate(y_true):
        onehot[i, int(t)] = 1
    # mean squared error between predicted probabilities and one-hot
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def load_models(model_dir='model'):
    md = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
    base = model_dir
    raw = joblib.load(os.path.join(base, md.get('raw_model', 'model_rf_raw.pkl')))
    # original raw model previously saved as model_rf.pkl may exist
    orig = None
    try:
        orig = joblib.load(os.path.join(base, 'model_rf.pkl'))
    except Exception:
        orig = raw
    calib = joblib.load(os.path.join(base, md.get('calibrated_model', 'model_calibrated.pkl')))
    feature_names = md.get('feature_names')
    return raw, orig, calib, feature_names


def main(data_path='..\\Fuzzy_Diabetes\\Model\\diabetes.csv'):
    os.makedirs('Paper', exist_ok=True)
    os.makedirs('Paper/figures', exist_ok=True)

    df = pd.read_csv(data_path)
    y = make_labels_from_glucose(df, 'Glucose')
    X = df.select_dtypes(include=[np.number]).copy()
    if 'Outcome' in X.columns:
        X = X.drop(columns=['Outcome'])
    X = X.fillna(X.median())

    raw, orig, calib, feature_names = load_models('model')

    # ensure proper columns
    if feature_names is not None:
        X_proc = X.reindex(columns=feature_names, fill_value=0.0)
    else:
        X_proc = X

    probs_raw = raw.predict_proba(X_proc)
    try:
        probs_orig = orig.predict_proba(X_proc)
    except Exception:
        probs_orig = probs_raw
    probs_cal = calib.predict_proba(X_proc)

    results = {}
    for name, probs in [('raw', probs_raw), ('orig', probs_orig), ('calibrated', probs_cal)]:
        acc = accuracy_score(y, np.argmax(probs, axis=1))
        prec, rec, f1, _ = precision_recall_fscore_support(y, np.argmax(probs, axis=1), average='macro', zero_division=0)
        brier = multiclass_brier_score(y.values, probs)
        results[name] = {'accuracy': float(acc), 'precision_macro': float(prec), 'recall_macro': float(rec), 'f1_macro': float(f1), 'brier': float(brier)}

    # save results table
    df_res = pd.DataFrame(results).T
    df_res.to_csv('Paper/results_table.csv')

    # calibration curves per class comparing raw vs calibrated
    n_classes = probs_cal.shape[1]
    for c in range(n_classes):
        plt.figure(figsize=(6,4))
        for label, probs in [('raw', probs_raw), ('calibrated', probs_cal)]:
            prob = probs[:, c]
            frac_pos, mean_pred = calibration_curve((y==c).astype(int), prob, n_bins=10)
            plt.plot(mean_pred, frac_pos, marker='o', label=f'{label}')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('Mean predicted'); plt.ylabel('Fraction of positives'); plt.title(f'Calibration class {c}')
        plt.legend(); plt.tight_layout()
        plt.savefig(f'Paper/figures/calibration_compare_class_{c}.png')

    # write markdown summary (avoid optional tabulate dependency)
    with open('Paper/results_summary.md', 'w', encoding='utf-8') as f:
        f.write('# Results summary\n\n')
        f.write('Comparison of raw RandomForest and calibrated classifier.\n\n')
        f.write('## Metrics table\n\n')
        f.write('|model|accuracy|precision_macro|recall_macro|f1_macro|brier|\n')
        f.write('|-|-|-|-|-|-|\n')
        for model_name, row in df_res.iterrows():
            f.write(f"|{model_name}|{row['accuracy']:.4f}|{row['precision_macro']:.4f}|{row['recall_macro']:.4f}|{row['f1_macro']:.4f}|{row['brier']:.6f}|\n")
        f.write('\n\n')
        f.write('## Calibration plots\n\n')
        for c in range(probs_cal.shape[1]):
            f.write(f'### Class {c}\n')
            f.write(f'![](figures/calibration_compare_class_{c}.png)\n\n')

    print('Saved Paper/results_table.csv and Paper/results_summary.md and calibration figures')


if __name__ == '__main__':
    main()
