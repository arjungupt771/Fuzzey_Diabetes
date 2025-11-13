"""Evaluate the saved model on the provided dataset and save reports/plots to model/.
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def make_labels_from_glucose(df, glucose_col='Glucose'):
    g = df[glucose_col].astype(float)
    labels = pd.Series(0, index=df.index)
    labels[(g >= 100) & (g < 126)] = 1
    labels[g >= 126] = 2
    return labels


def main(data_path="..\\Fuzzy_Diabetes\\Model\\diabetes.csv", model_path="model\\model_rf.pkl"):
    if not os.path.exists(model_path):
        raise SystemExit(f'Model not found at {model_path}. Train first.')

    m = joblib.load(model_path)
    model = m.get('model') if isinstance(m, dict) else m
    feature_names = m.get('feature_names', None) if isinstance(m, dict) else None

    df = pd.read_csv(data_path)
    y = make_labels_from_glucose(df, 'Glucose')
    X = df.select_dtypes(include=[np.number]).copy()
    if 'Outcome' in X.columns:
        X = X.drop(columns=['Outcome'])
    X = X.fillna(X.median())

    preds = model.predict(X)

    report = classification_report(y, preds, digits=4)
    print(report)
    os.makedirs('model', exist_ok=True)
    with open('model/eval_report.txt', 'w') as f:
        f.write(report)

    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (0=normal,1=prediabetic,2=diabetic)')
    plt.tight_layout()
    plt.savefig('model/confusion_matrix.png')
    print('Saved model/eval_report.txt and model/confusion_matrix.png')


if __name__ == '__main__':
    main()
