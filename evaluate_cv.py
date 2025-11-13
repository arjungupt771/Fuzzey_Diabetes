"""Cross-validated evaluation producing ROC/PR, calibration, and SHAP summary plots.
Saves figures to Paper/figures.
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.calibration import calibration_curve


def make_labels_from_glucose(df, glucose_col='Glucose'):
    g = df[glucose_col].astype(float)
    labels = pd.Series(0, index=df.index)
    labels[(g >= 100) & (g < 126)] = 1
    labels[g >= 126] = 2
    return labels


def main(data_path='..\\Fuzzy_Diabetes\\Model\\diabetes.csv', out_dir='Paper/figures'):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    y = make_labels_from_glucose(df, 'Glucose')
    X = df.select_dtypes(include=[np.number]).copy()
    if 'Outcome' in X.columns:
        X = X.drop(columns=['Outcome'])
    X = X.fillna(X.median())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_classes = 3

    # collect probabilities and true labels
    all_probs = np.zeros((len(X), n_classes))
    all_true = np.array(y)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        probs = rf.predict_proba(X_test)
        all_probs[test_idx, :] = probs

    # ROC and PR for each class (one-vs-rest)
    plt.figure(figsize=(8,6))
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve((all_true==c).astype(int), all_probs[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'class {c} (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC per class'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'roc_per_class.png'))

    # PR curves
    plt.figure(figsize=(8,6))
    for c in range(n_classes):
        prec, rec, _ = precision_recall_curve((all_true==c).astype(int), all_probs[:, c])
        ap = average_precision_score((all_true==c).astype(int), all_probs[:, c])
        plt.plot(rec, prec, label=f'class {c} (AP={ap:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR per class'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'pr_per_class.png'))

    # Calibration curves per class
    plt.figure(figsize=(8,6))
    for c in range(n_classes):
        prob = all_probs[:, c]
        fraction_of_positives, mean_predicted_value = calibration_curve((all_true==c).astype(int), prob, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'class {c}')
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('Mean predicted value'); plt.ylabel('Fraction of positives'); plt.title('Calibration per class'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'calibration_per_class.png'))

    # SHAP summary using a final RandomForest trained on all data
    try:
        import shap
        rf_final = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_final.fit(X, y)
        expl = shap.TreeExplainer(rf_final)
        shap_vals = expl.shap_values(X)
        # summary plot for top class 2 as example
        plt.figure()
        shap.summary_plot(shap_vals, X, show=False)
        plt.title('SHAP summary (all classes)')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'shap_summary.png'))
    except Exception:
        print('SHAP not available or failed; skipping SHAP plot')

    print('Saved figures to', out_dir)


if __name__ == '__main__':
    main()
