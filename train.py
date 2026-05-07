import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from visualize import EnergyVisualizer
from data_cleaner import DataCleaner
from data_loader import SequenceDataLoader
from model import build_lstm_model, compile_model
import tensorflow as tf
import numpy as np


class IndiaEnergyTrainer:
    def __init__(self, seq_length=7, epochs=100, batch_size=16):
        """
        seq_length=7  : 7-day lookback (weekly pattern)
        batch_size=16 : small — we have few sequences, need small batches
        epochs=100    : more epochs since dataset is tiny and each epoch is fast
        """
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.trained_models = {}
        self.training_history = {}

    def run_full_pipeline(self):
        print("\n" + "="*70)
        print("STEP 1: DATA CLEANING")
        print("="*70)
        cleaner = DataCleaner(data_dir='data')
        cleaner.run()
        cleaner.print_report()
        cleaner.save_cleaned_data()

        print("\n" + "="*70)
        print("STEP 2: SEQUENCE PREPARATION  (7-day lookback)")
        print("="*70)
        loader = SequenceDataLoader(
            cleaned_data_dir='cleaned_data',
            seq_length=self.seq_length
        )
        loader.load_cleaned_data()
        prepared_data = loader.prepare_all_regions()

        print("\n" + "="*70)
        print("STEP 3: MODEL TRAINING")
        print("="*70)
        self.train_all_models(prepared_data)

        print("\n" + "="*70)
        print("STEP 4: SAVING MODELS")
        print("="*70)
        self.save_all_models()

        print("\n✓ PIPELINE COMPLETE!")
        return prepared_data

    def is_collapsed(self, history):
        """Detect if model predicted constant (all-zero loss is a red flag)"""
        final_mae = history.history['mae'][-1]
        return final_mae < 1e-7

    def train_model(self, state_name, data):
        if data is None:
            return

        X_train = data['X_train'].reshape(-1, self.seq_length, 1)
        X_test  = data['X_test'].reshape(-1, self.seq_length, 1)
        y_train = data['y_train']
        y_test  = data['y_test']

        print(f"\nTraining {state_name}...")
        print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

        # Smaller model for small dataset — large model = instant overfit
        model = build_lstm_model(
            seq_length=self.seq_length,
            lstm_units=32,        # was 64 — reduced for small data
            dropout_rate=0.1      # less dropout — fewer params to regularize
        )
        model = compile_model(model, learning_rate=0.001)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,              # more patience for small dataset
                restore_best_weights=True,
                verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, patience=7,
                min_lr=1e-6, verbose=0
            ),
        ]

        history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Check for collapse
        if self.is_collapsed(history):
            print(f"  ⚠ Model COLLAPSED for {state_name} — predicting constant")
            print(f"     This state may have too little data variation to learn from")
            status = "⚠ COLLAPSED"
        else:
            train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
            test_loss,  test_mae  = model.evaluate(X_test,  y_test,  verbose=0)
            status = "✓ GOOD" if test_mae < 0.1 else "~ OK"
            print(f"\n  {status} {state_name}: Train MAE={train_mae:.4f}  Test MAE={test_mae:.4f}")

            self.trained_models[state_name] = {
                'model':      model,
                'train_loss': model.evaluate(X_train, y_train, verbose=0)[0],
                'train_mae':  model.evaluate(X_train, y_train, verbose=0)[1],
                'test_loss':  model.evaluate(X_test, y_test, verbose=0)[0],
                'test_mae':   model.evaluate(X_test, y_test, verbose=0)[1],
                'X_test':     X_test,
                'y_test':     y_test
            }
            self.training_history[state_name] = history

    def train_all_models(self, prepared_data):
        total = len(prepared_data)
        for i, (state_name, data) in enumerate(prepared_data.items(), 1):
            print(f"\n[{i}/{total}] ── {state_name} " + "─"*30)
            self.train_model(state_name, data)
            tf.keras.backend.clear_session()

    def save_all_models(self):
        os.makedirs('models', exist_ok=True)
        for state_name, d in self.trained_models.items():
            path = f"models/{state_name}_lstm.h5"
            d['model'].save(path)
            print(f"✓ {state_name} → {path}")

    def print_training_summary(self):
        print("\n" + "="*70)
        print("INDIA ENERGY TRAINING SUMMARY")
        print("="*70)
        print(f"{'State':<25} {'Train MAE':>10} {'Test MAE':>10} {'Status'}")
        print("-"*65)
        for state, d in self.trained_models.items():
            status = "✓ GOOD" if d['test_mae'] < 0.1 else "~ OK"
            print(f"{state:<25} {d['train_mae']:>10.4f} {d['test_mae']:>10.4f}  {status}")

        if self.trained_models:
            visualizer = EnergyVisualizer(self.trained_models, self.training_history)
            visualizer.generate_all_plots()
        else:
            print("\n⚠ No models trained successfully — check your data!")


if __name__ == "__main__":
    trainer = IndiaEnergyTrainer(
        seq_length=7,    # 7-day lookback
        epochs=100,      # more epochs, each one is fast
        batch_size=16    # small batch for small dataset
    )
    prepared_data = trainer.run_full_pipeline()
    trainer.print_training_summary()

# conda activate tf-apple