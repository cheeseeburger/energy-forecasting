import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.cleaned_data = {}
        self.scalers = {}
        self.cleaning_report = {}

    def load_india_data(self):
        """
        Reads both CSVs with their EXACT confirmed structure:

        dataset_tk.csv:
            col[0] = 'Unnamed: 0'  → datetime strings  (date column)
            col[1..33]              → state names       (power values in MW)

        long_data_.csv:
            'States'  → state name
            'Dates'   → datetime string
            'Usage'   → power in MW
            (also has Regions, latitude, longitude — ignored)
        """
        state_data = {}

        # ── FILE 1: dataset_tk.csv (wide format) ──────────────────────────
        tk_path = os.path.join(self.data_dir, 'dataset_tk.csv')
        if os.path.exists(tk_path):
            df = pd.read_csv(tk_path)
            print(f"✓ dataset_tk.csv  ({df.shape[0]} rows × {df.shape[1]} cols)")

            # Column 0 is the date ('Unnamed: 0')
            date_col = df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.rename(columns={date_col: 'Datetime'})

            # Every other column is a state
            state_cols = [c for c in df.columns if c != 'Datetime']
            print(f"  States found: {state_cols}")

            for col in state_cols:
                sdf = df[['Datetime', col]].copy()
                sdf = sdf.rename(columns={col: 'Power_consumed'})
                sdf['Power_consumed'] = pd.to_numeric(sdf['Power_consumed'], errors='coerce')
                sdf = sdf.dropna()
                sdf = sdf.sort_values('Datetime').reset_index(drop=True)
                key = col.strip().replace(' ', '_')
                state_data[key] = sdf
                print(f"    + {key}: {len(sdf)} rows | "
                      f"range {sdf['Power_consumed'].min():.1f}–{sdf['Power_consumed'].max():.1f} MW")
        else:
            print("⚠ dataset_tk.csv not found")

        # ── FILE 2: long_data_.csv (long format) ──────────────────────────
        long_path = os.path.join(self.data_dir, 'long_data_.csv')
        if os.path.exists(long_path):
            df2 = pd.read_csv(long_path)
            print(f"\n✓ long_data_.csv  ({df2.shape[0]} rows × {df2.shape[1]} cols)")

            df2['Dates'] = pd.to_datetime(df2['Dates'], dayfirst=True, errors='coerce')
            df2 = df2.dropna(subset=['Dates', 'Usage'])

            for state, grp in df2.groupby('States'):
                key = state.strip().replace(' ', '_')
                sdf = grp[['Dates', 'Usage']].copy()
                sdf = sdf.rename(columns={'Dates': 'Datetime', 'Usage': 'Power_consumed'})
                sdf = sdf.sort_values('Datetime').reset_index(drop=True)

                if key in state_data:
                    # Merge with dataset_tk data (deduplicate by date)
                    combined = pd.concat([state_data[key], sdf])
                    combined = combined.drop_duplicates(subset=['Datetime'])
                    combined = combined.sort_values('Datetime').reset_index(drop=True)
                    state_data[key] = combined
                    print(f"    ~ {key}: merged → {len(combined)} rows total")
                else:
                    state_data[key] = sdf
                    print(f"    + {key}: {len(sdf)} rows")
        else:
            print("⚠ long_data_.csv not found")

        print(f"\n{'='*55}")
        print(f"Total Indian states loaded: {len(state_data)}")
        return state_data

    def clean_region(self, df, region_name):
        initial_rows = len(df)
        energy_col = 'Power_consumed'

        # Verify data actually has variance BEFORE normalizing
        raw_std = df[energy_col].std()
        if raw_std < 1e-6:
            print(f"  ✗ {region_name}: zero variance in raw data ({df[energy_col].unique()[:3]}) — skipping")
            return None

        # Missing values
        missing = int(df.isnull().sum().sum())
        df = df.ffill(limit=7).bfill()

        # Duplicates
        dup_count = int(df.duplicated(subset=['Datetime']).sum())
        df = df.drop_duplicates(subset=['Datetime'])
        df = df.sort_values('Datetime').reset_index(drop=True)

        # Outliers
        Q1, Q3 = df[energy_col].quantile(0.25), df[energy_col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outlier_count = int(((df[energy_col] < lower) | (df[energy_col] > upper)).sum())
        df[energy_col] = df[energy_col].clip(lower=lower, upper=upper)

        # Per-state normalization
        scaler = MinMaxScaler()
        df['Power_consumed_normalized'] = scaler.fit_transform(
            df[energy_col].values.reshape(-1, 1)
        )
        self.scalers[region_name] = scaler

        # Verify normalization has variance
        norm_std = df['Power_consumed_normalized'].std()
        if norm_std < 1e-6:
            print(f"  ✗ {region_name}: normalized data has zero variance — skipping")
            return None

        self.cleaning_report[region_name] = {
            'initial_rows': initial_rows,
            'final_rows': len(df),
            'missing_values': missing,
            'duplicates_removed': dup_count,
            'outliers_clipped': outlier_count,
            'value_range': f"{df[energy_col].min():.1f}–{df[energy_col].max():.1f} MW",
            'norm_std': f"{norm_std:.4f}",
            'date_range': f"{df['Datetime'].min().date()} → {df['Datetime'].max().date()}"
        }

        print(f"  ✓ {region_name:<20} {len(df):>4} rows | "
              f"{df[energy_col].min():.0f}–{df[energy_col].max():.0f} MW | "
              f"norm_std={norm_std:.4f}")
        return df

    def run(self):
        print("\n" + "="*60)
        print("INDIA STATE ENERGY — DATA CLEANING")
        print("="*60)
        state_data = self.load_india_data()

        print("\nCleaning...")
        for region_name, df in state_data.items():
            result = self.clean_region(df, region_name)
            if result is not None:
                self.cleaned_data[region_name] = result

        print(f"\n✓ {len(self.cleaned_data)} states cleaned and ready")
        return self.cleaned_data

    def save_cleaned_data(self, output_dir='cleaned_data'):
        os.makedirs(output_dir, exist_ok=True)
        for region_name, df in self.cleaned_data.items():
            df.to_csv(os.path.join(output_dir, f"{region_name}_cleaned.csv"), index=False)
        print(f"✓ Saved to /{output_dir}")

    def print_report(self):
        print("\n" + "="*75)
        print(f"{'State':<22} {'Rows':>5} {'Missing':>7} {'Dups':>5} {'Outliers':>9}  Range")
        print("-"*75)
        for r, s in self.cleaning_report.items():
            print(f"{r:<22} {s['final_rows']:>5} {s['missing_values']:>7} "
                  f"{s['duplicates_removed']:>5} {s['outliers_clipped']:>9}  {s['value_range']}")


if __name__ == "__main__":
    cleaner = DataCleaner(data_dir='data')
    cleaner.run()
    cleaner.print_report()
    cleaner.save_cleaned_data()