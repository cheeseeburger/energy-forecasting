import pandas as pd
import numpy as np
import os


class SequenceDataLoader:
    def __init__(self, cleaned_data_dir='cleaned_data', seq_length=7):
        """
        seq_length=7 for daily Indian data with ~498 rows.
        
        Why 7 and not 30?
        - 498 rows - 30 seq = only 468 sequences → model collapses (predicts mean)
        - 498 rows - 7 seq  = 491 sequences → enough variation to actually learn
        - 7 days lookback also makes real sense: weekly energy patterns repeat
        """
        self.cleaned_data_dir = cleaned_data_dir
        self.seq_length = seq_length
        self.data_dict = {}

    def load_cleaned_data(self):
        files = [f for f in os.listdir(self.cleaned_data_dir) if f.endswith('.csv')]
        for file in sorted(files):
            try:
                df = pd.read_csv(os.path.join(self.cleaned_data_dir, file))
                region_name = file.replace('_cleaned.csv', '')
                self.data_dict[region_name] = df
                print(f"✓ Loaded {region_name}  ({len(df)} rows)")
            except Exception as e:
                print(f"✗ Failed {file}: {e}")
        return self.data_dict

    def create_sequences(self, data):
        """
        Input:  [day t-7, t-6, t-5, t-4, t-3, t-2, t-1]
        Output: [day t]
        """
        if 'Power_consumed_normalized' in data.columns:
            values = data['Power_consumed_normalized'].values
        elif 'Power_consumed' in data.columns:
            values = data['Power_consumed'].values
        else:
            values = data.iloc[:, 1].values

        # Sanity check — if all values are the same, skip
        if np.std(values) < 1e-6:
            print(f"  ⚠ Zero variance in data — all values identical, skipping")
            return None, None

        X, y = [], []
        for i in range(len(values) - self.seq_length):
            X.append(values[i:i + self.seq_length])
            y.append(values[i + self.seq_length])

        return np.array(X), np.array(y)

    def train_test_split(self, X, y, train_ratio=0.8):
        split = int(len(X) * train_ratio)
        return X[:split], X[split:], y[:split], y[split:]

    def prepare_region_data(self, region_name):
        if region_name not in self.data_dict:
            return None

        df = self.data_dict[region_name]
        print(f"\nPreparing {region_name}  ({len(df)} rows, seq_length={self.seq_length})")

        X, y = self.create_sequences(df)

        if X is None:
            return None

        if len(X) < 30:
            print(f"  ⚠ Only {len(X)} sequences — too few, skipping")
            return None

        X_train, X_test, y_train, y_test = self.train_test_split(X, y)
        print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

        # Warn if training set is dangerously small
        if len(X_train) < 50:
            print(f"  ⚠ WARNING: only {len(X_train)} training sequences — results may be poor")

        return {
            'X_train': X_train,
            'X_test':  X_test,
            'y_train': y_train,
            'y_test':  y_test,
        }

    def prepare_all_regions(self):
        prepared = {}
        for region_name in self.data_dict.keys():
            result = self.prepare_region_data(region_name)
            if result is not None:
                prepared[region_name] = result
        print(f"\n✓ {len(prepared)} states ready for training")
        return prepared


if __name__ == "__main__":
    loader = SequenceDataLoader(seq_length=7)
    loader.load_cleaned_data()
    data = loader.prepare_all_regions()