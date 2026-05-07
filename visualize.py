import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os


# Indian state groupings by region
INDIA_REGIONS = {
    'North':     ['Punjab', 'Haryana', 'Rajasthan', 'Delhi', 'Himachal_Pradesh', 'Jammu_and_Kashmir', 'Uttarakhand', 'Uttar_Pradesh'],
    'South':     ['Tamil_Nadu', 'Karnataka', 'Andhra_Pradesh', 'Telangana', 'Kerala'],
    'West':      ['Maharashtra', 'Gujarat', 'Goa', 'Madhya_Pradesh'],
    'East':      ['West_Bengal', 'Odisha', 'Bihar', 'Jharkhand', 'Chhattisgarh'],
    'Northeast': ['Assam', 'Meghalaya', 'Manipur', 'Nagaland', 'Tripura', 'Mizoram', 'Arunachal_Pradesh', 'Sikkim'],
}

REGION_COLORS = {
    'North': '#1976D2',
    'South': '#388E3C',
    'West':  '#F57C00',
    'East':  '#7B1FA2',
    'Northeast': '#0097A7',
    'Other': '#757575'
}


def get_region(state_name):
    for region, states in INDIA_REGIONS.items():
        if state_name in states:
            return region
    return 'Other'


class EnergyVisualizer:
    def __init__(self, trained_models, training_history):
        self.trained_models = trained_models
        self.training_history = training_history
        os.makedirs('plots', exist_ok=True)

    # ─────────────────────────────────────────────
    # 1. Actual vs Predicted
    # ─────────────────────────────────────────────
    def plot_predictions(self, state_name):
        d = self.trained_models[state_name]
        model  = d['model']
        X_test = d['X_test']
        y_test = d['y_test']

        predictions = model.predict(X_test, verbose=0).flatten()
        n = min(len(y_test), 120)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'{state_name.replace("_"," ")} — Power Consumption Forecast',
                     fontsize=14, fontweight='bold')

        # Full test set
        axes[0].plot(y_test, label='Actual', color='#1976D2', linewidth=1, alpha=0.8)
        axes[0].plot(predictions, label='Predicted', color='#E53935',
                     linewidth=1, linestyle='--', alpha=0.85)
        axes[0].set_title('Full test set')
        axes[0].set_ylabel('Normalized power')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Zoomed 120 days
        axes[1].plot(y_test[:n], label='Actual', color='#1976D2', linewidth=1.5)
        axes[1].plot(predictions[:n], label='Predicted', color='#E53935',
                     linewidth=1.5, linestyle='--')
        axes[1].fill_between(range(n),
                              y_test[:n], predictions[:n],
                              alpha=0.15, color='#E53935', label='Error')
        axes[1].set_title(f'Zoomed: first {n} days')
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('Normalized power')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'plots/{state_name}_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {state_name}_predictions.png")

    # ─────────────────────────────────────────────
    # 2. Training loss curve
    # ─────────────────────────────────────────────
    def plot_training_loss(self, state_name):
        history = self.training_history[state_name]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f'{state_name.replace("_"," ")} — Training Curves', fontweight='bold')

        ax1.plot(history.history['loss'],     label='Train loss', color='#1976D2', linewidth=1.5)
        ax1.plot(history.history['val_loss'], label='Val loss',   color='#F57C00', linewidth=1.5)
        ax1.set_title('MSE Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history.history['mae'],     label='Train MAE', color='#388E3C', linewidth=1.5)
        ax2.plot(history.history['val_mae'], label='Val MAE',   color='#E53935', linewidth=1.5)
        ax2.set_title('Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'plots/{state_name}_loss.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {state_name}_loss.png")

    # ─────────────────────────────────────────────
    # 3. All-India MAE comparison (color by region)
    # ─────────────────────────────────────────────
    def plot_india_comparison(self):
        states    = list(self.trained_models.keys())
        test_maes = [self.trained_models[s]['test_mae']  for s in states]
        train_maes= [self.trained_models[s]['train_mae'] for s in states]
        colors    = [REGION_COLORS[get_region(s)] for s in states]

        # Sort by test MAE
        order = np.argsort(test_maes)
        states     = [states[i]     for i in order]
        test_maes  = [test_maes[i]  for i in order]
        train_maes = [train_maes[i] for i in order]
        colors     = [colors[i]     for i in order]

        fig, ax = plt.subplots(figsize=(max(12, len(states) * 0.7), 6))

        x = np.arange(len(states))
        w = 0.35
        bars1 = ax.bar(x - w/2, train_maes, w, label='Train MAE',
                       color=colors, alpha=0.6, edgecolor='white')
        bars2 = ax.bar(x + w/2, test_maes,  w, label='Test MAE',
                       color=colors, alpha=1.0, edgecolor='white')

        ax.axhline(y=0.05, color='green', linestyle='--', linewidth=1, label='Target (0.05)')
        ax.set_title('India State-wise LSTM Forecast Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('State')
        ax.set_ylabel('MAE (normalized)')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in states], fontsize=8, rotation=0)

        # Value labels
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

        # Region legend
        legend_patches = [mpatches.Patch(color=c, label=r)
                          for r, c in REGION_COLORS.items()
                          if r in [get_region(s) for s in states]]
        legend_patches.append(plt.Line2D([0], [0], color='green', linestyle='--', label='Target'))
        ax.legend(handles=legend_patches, loc='upper left', fontsize=9)

        plt.tight_layout()
        plt.savefig('plots/india_all_states_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ india_all_states_comparison.png")

    # ─────────────────────────────────────────────
    # 4. Regional heatmap (avg consumption by month)
    # ─────────────────────────────────────────────
    def plot_monthly_heatmap(self):
        """Load cleaned data and show monthly average power by state"""
        if not os.path.exists('cleaned_data'):
            print("  ⚠ No cleaned_data folder found, skipping heatmap")
            return

        files = [f for f in os.listdir('cleaned_data') if f.endswith('_cleaned.csv')]
        monthly_data = {}

        for file in files:
            state = file.replace('_cleaned.csv', '')
            if state not in self.trained_models:
                continue
            try:
                df = pd.read_csv(os.path.join('cleaned_data', file), parse_dates=['Datetime'])
                df['Month'] = df['Datetime'].dt.month
                monthly_avg = df.groupby('Month')['Power_consumed'].mean()
                monthly_data[state.replace('_', ' ')] = monthly_avg
            except:
                pass

        if len(monthly_data) < 2:
            print("  ⚠ Not enough data for heatmap")
            return

        heatmap_df = pd.DataFrame(monthly_data).T
        heatmap_df.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                               'Jul','Aug','Sep','Oct','Nov','Dec'][:len(heatmap_df.columns)]

        fig, ax = plt.subplots(figsize=(14, max(6, len(heatmap_df) * 0.4)))
        im = ax.imshow(heatmap_df.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(heatmap_df.columns)))
        ax.set_xticklabels(heatmap_df.columns)
        ax.set_yticks(range(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index, fontsize=9)
        ax.set_title('Monthly Average Power Consumption by Indian State (MW)',
                     fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Avg MW')
        plt.tight_layout()
        plt.savefig('plots/india_monthly_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ india_monthly_heatmap.png")

    # ─────────────────────────────────────────────
    # 5. Top vs Bottom performers
    # ─────────────────────────────────────────────
    def plot_best_worst(self):
        states    = list(self.trained_models.keys())
        test_maes = [self.trained_models[s]['test_mae'] for s in states]
        sorted_idx = np.argsort(test_maes)
        n = min(5, len(states) // 2)

        best  = [(states[i], test_maes[i]) for i in sorted_idx[:n]]
        worst = [(states[i], test_maes[i]) for i in sorted_idx[-n:]]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('India LSTM — Best vs Worst Performing States', fontweight='bold')

        for ax, group, color, title in [
            (axes[0], best,  '#388E3C', f'Top {n} States (lowest MAE)'),
            (axes[1], worst, '#E53935', f'Bottom {n} States (highest MAE)')
        ]:
            names = [s.replace('_', '\n') for s, _ in group]
            maes  = [m for _, m in group]
            bars  = ax.barh(names, maes, color=color, alpha=0.8, edgecolor='white')
            ax.set_title(title)
            ax.set_xlabel('Test MAE')
            ax.axvline(x=0.05, color='black', linestyle='--', linewidth=1, label='Target')
            ax.legend()
            for bar, val in zip(bars, maes):
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=9)
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('plots/india_best_worst_states.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ india_best_worst_states.png")

    # ─────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────
    def generate_all_plots(self):
        print("\n" + "="*50)
        print("GENERATING INDIA VISUALIZATIONS")
        print("="*50)

        for state_name in self.trained_models.keys():
            print(f"\nPlotting {state_name}...")
            try:
                self.plot_predictions(state_name)
                self.plot_training_loss(state_name)
            except Exception as e:
                print(f"  ✗ Error: {e}")

        print("\nGenerating summary plots...")
        try: self.plot_india_comparison()
        except Exception as e: print(f"  ✗ Comparison: {e}")

        try: self.plot_monthly_heatmap()
        except Exception as e: print(f"  ✗ Heatmap: {e}")

        try: self.plot_best_worst()
        except Exception as e: print(f"  ✗ Best/worst: {e}")

        print(f"\n✓ All plots saved to /plots")


if __name__ == "__main__":
    print("Run train.py first — visualizations auto-generate at the end!")