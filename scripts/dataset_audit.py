"""Create a compact data-quality and subgroup audit for a diabetes CSV."""

import argparse
import os

import numpy as np
import pandas as pd

from diabetes_risk.ml_pipeline import make_target, select_features


def audit_dataset(data_path, output_path='Paper/dataset_audit.md', target_column=None):
    df = pd.read_csv(data_path)
    if target_column is None and 'Glucose' not in df.columns:
        raise ValueError('Dataset must contain Glucose to derive the prototype research labels')
    y, label_names, target_definition, target_column = make_target(df, target_column)
    X = select_features(df, target_column if target_definition.startswith('independent') else None)
    missing = df.isna().sum()
    numeric = df.select_dtypes(include=[np.number])
    duplicate_count = int(df.duplicated().sum())
    class_counts = y.value_counts().reindex(label_names, fill_value=0)
    lines = [
        '# Dataset quality audit', '',
        f'- Source: `{os.path.abspath(data_path)}`',
        f'- Rows: {len(df)}; columns: {len(df.columns)}',
        f'- Exact duplicate rows: {duplicate_count}',
        f'- Predictor columns after leakage exclusions: {", ".join(X.columns)}',
        f'- Target definition: {target_definition}',
        f'- Target-defining column excluded from predictors: {target_column}', '',
        '## Class counts', '', '|class|count|share|', '|---|---:|---:|',
    ]
    for label, count in class_counts.items():
        lines.append(f'|{label_names[int(label)]}|{int(count)}|{count / max(len(y), 1):.3f}|')
    lines += ['', '## Missing values', '', '|column|missing|share|', '|---|---:|---:|']
    for column, count in missing[missing > 0].sort_values(ascending=False).items():
        lines.append(f'|{column}|{int(count)}|{count / max(len(df), 1):.3f}|')
    if not missing.any():
        lines.append('|None|0|0.000|')
    lines += ['', '## Numeric feature ranges and IQR outliers', '',
              '|feature|missing|minimum|median|maximum|IQR outlier rows|',
              '|---|---:|---:|---:|---:|---:|']
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        outliers = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
        lines.append(f'|{column}|{int(numeric[column].isna().sum())}|{values.min():.4g}|'
                     f'{values.median():.4g}|{values.max():.4g}|{outliers}|')
    for feature, bins in [('Age', [0, 35, 50, np.inf]), ('BMI', [0, 25, 30, np.inf])]:
        if feature in df.columns:
            groups = pd.cut(df[feature], bins=bins, right=False, include_lowest=True)
            grouped = pd.DataFrame({'group': groups, 'class': y}).groupby(
                ['group', 'class'], observed=False
            ).size().unstack(fill_value=0)
            lines += ['', f'## Label counts by {feature}', '',
                      '|group|' + '|'.join(label_names[int(c)] for c in grouped.columns) + '|',
                      '|---|' + '|'.join('---:' for _ in grouped.columns) + '|']
            for group, counts in grouped.iterrows():
                lines.append('|' + str(group) + '|' + '|'.join(str(int(v)) for v in counts) + '|')
    lines += ['', '## Interpretation limits', '',
              '- IQR outliers are flags for review, not evidence of invalid records.',
              '- Age/BMI tables describe this dataset and its labels; they do not establish fairness or subgroup model performance.',
              '- Duplicate detection here checks exact duplicate rows only.',
              '- Independent cohort validation and label provenance are still required.', '']
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f'Wrote audit to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', default='Paper/dataset_audit.md')
    parser.add_argument('--target-column', default=None, help='Independent 3-class target column')
    args = parser.parse_args()
    audit_dataset(args.data, args.out, args.target_column)
