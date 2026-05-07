import pandas as pd
import numpy as np
import os

print("="*60)
print("DIAGNOSING CSV FILES IN /data")
print("="*60)

for file in os.listdir('data'):
    if not file.endswith('.csv'):
        continue
    path = f'data/{file}'
    print(f"\n{'='*60}")
    print(f"FILE: {file}")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] '{col}'")

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nSample values per column:")
    for col in df.columns:
        vals = df[col].dropna().unique()[:5]
        print(f"  '{col}': {list(vals)}")

    print(f"\nNull counts:")
    print(df.isnull().sum())

    print(f"\nNumeric columns stats:")
    numeric = df.select_dtypes(include='number')
    if not numeric.empty:
        print(numeric.describe())
    else:
        print("  No numeric columns found!")