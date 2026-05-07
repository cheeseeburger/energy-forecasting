# India State-wise Energy Consumption Forecasting
### LSTM Deep Learning Model — 33 Indian States

---

## Project Overview

This project builds and trains an LSTM (Long Short-Term Memory) neural network to forecast daily power consumption for all 33 Indian states and union territories. Given the last 7 days of electricity usage, the model predicts the next day's consumption — a real-world problem used by power grid operators to balance supply and demand across India's five regional grids.

**Dataset:** Kaggle — State-wise Power Consumption in India (2019–2020)  
**Model:** Stacked LSTM with BatchNormalization and gradient clipping  
**Coverage:** 33 states including Maharashtra, UP, Rajasthan, Tamil Nadu, Delhi and more  
**Result:** MAE of 0.13–0.25 (normalized) across all states

---

## Project Structure

```
energy_forecasting/
│
├── data/
│   ├── dataset_tk.csv          # Wide format: Date × 33 states (MW values)
│   └── long_data_.csv          # Long format: States | Regions | Dates | Usage
│
├── cleaned_data/               # Auto-generated cleaned CSVs per state
│
├── models/                     # Saved .h5 LSTM models per state
│
├── plots/                      # All visualizations (auto-generated)
│   ├── {State}_predictions.png     # Actual vs Predicted per state
│   ├── {State}_loss.png            # Training loss curves per state
│   ├── india_all_states_comparison.png  # MAE comparison across all states
│   ├── india_monthly_heatmap.png        # Monthly consumption heatmap
│   └── india_best_worst_states.png      # Top vs bottom performing states
│
├── data_cleaner.py             # Data cleaning pipeline
├── data_loader.py              # Sequence generation for LSTM
├── model.py                    # LSTM architecture definition
├── train.py                    # Full training pipeline
└── visualize.py                # All visualization generation
```

---

## How It Works

### Step 1 — Data Cleaning (`data_cleaner.py`)
- Loads both CSV formats (wide + long) and parses all 33 states
- Handles missing values with forward-fill (max 7 days)
- Removes duplicate timestamps
- Clips outliers using IQR method
- Applies **per-state MinMax normalization** to [0, 1] — critical because Maharashtra uses 400+ MW while Sikkim uses only 2 MW

### Step 2 — Sequence Generation (`data_loader.py`)
- Creates sliding window sequences of length 7 (7-day lookback)
- Input: `[day t-7, t-6, t-5, t-4, t-3, t-2, t-1]` → Output: `[day t]`
- Splits 80% train / 20% test (time-ordered, no shuffling)

### Step 3 — LSTM Model (`model.py`)
```
Input (7 timesteps × 1 feature)
    ↓
LSTM Layer 1 (32 units, tanh activation)
    ↓
BatchNormalization
    ↓
Dropout (10%)
    ↓
LSTM Layer 2 (16 units, tanh activation)
    ↓
BatchNormalization
    ↓
Dense (8 units, relu)
    ↓
Output (1 value — next day's normalized consumption)
```

**Key design choices:**
- `tanh` activation (not relu) — relu causes gradient explosions inside LSTM gates
- `BatchNormalization` — stabilizes training across states with very different MW scales
- `clipnorm=1.0` in Adam optimizer — prevents loss from exploding to billions (seen in early testing with FE/AP regions)
- `legacy.Adam` — fixes Mac M1/M2 slowdown warnings

### Step 4 — Training (`train.py`)
- Trains one model per state sequentially
- Early stopping (patience=15) stops training when validation loss stops improving
- ReduceLROnPlateau halves the learning rate when stuck (patience=7)
- Clears Keras session between states to free memory

### Step 5 — Visualization (`visualize.py`)
Auto-generates 5 types of plots after training completes:

| Plot | What it shows |
|---|---|
| `{State}_predictions.png` | Actual vs LSTM predicted consumption (full + zoomed view) |
| `{State}_loss.png` | MSE loss + MAE curves over training epochs |
| `india_all_states_comparison.png` | All 33 states ranked by Test MAE, color-coded by region |
| `india_monthly_heatmap.png` | Average MW consumption by state × month |
| `india_best_worst_states.png` | Top 5 vs bottom 5 performing states |

---

## Understanding the Results

### MAE Score Interpretation
Since all data is normalized to [0, 1]:

| MAE Range | Meaning |
|---|---|
| < 0.10 | Excellent prediction |
| 0.10 – 0.20 | Good — model capturing trends |
| 0.20 – 0.30 | Acceptable — some variation missed |
| > 0.30 | Poor — model struggling |

Most states achieved 0.13–0.20 which is good given only 498 days of training data.

### Reading the Prediction Plots
- **Blue solid line** = actual real power consumption (ground truth)
- **Red dashed line** = LSTM model's prediction
- **Pink shaded area** = the error gap between prediction and reality
- The model captures long-term trends well; sharp single-day spikes are harder to predict with 7-day lookback

### Reading the Loss Curves
- **Train loss going down** = model is learning ✅
- **Val loss going down alongside train** = model generalizes well, not memorizing ✅
- **Val loss going up while train goes down** = overfitting (early stopping handles this)
- Both curves converging flat = model has reached its learning limit

### Why Some States Have Higher MAE
States like Manipur (MAE ~0.21) and Chandigarh have higher error because:
- Smaller absolute consumption (2–5 MW) means any variation is proportionally large
- More irregular usage patterns (less predictable weekly cycles)
- Same 498 rows of data but higher noise-to-signal ratio

---

## Setup and Usage

### Prerequisites
```bash
conda activate tf-apple   # Mac M1/M2
# or your TensorFlow environment

pip install tensorflow scikit-learn pandas numpy matplotlib
```

### Run the Full Pipeline
```bash
python train.py
```
This automatically runs all 5 steps: clean → load → train → save → visualize.

### Run Individual Steps
```bash
python data_cleaner.py    # Just clean the data
python data_loader.py     # Check sequence generation
python model.py           # Print model architecture summary
```

---

## Dataset

**Source:** [Kaggle — State-wise Power Consumption in India](https://www.kaggle.com/datasets/)

Two CSV files:
- `dataset_tk.csv` — 503 rows × 34 columns (wide format, one column per state)
- `long_data_.csv` — 16,599 rows × 6 columns (long format with lat/lon/region metadata)

**Coverage:** February 2019 – June 2020 (daily readings)  
**Unit:** MW (Megawatts) of power consumed per day  
**States covered:** Punjab, Haryana, Rajasthan, Delhi, UP, Uttarakhand, HP, J&K, Chandigarh, Chhattisgarh, Gujarat, MP, Maharashtra, Goa, DNH, Andhra Pradesh, Telangana, Karnataka, Kerala, Tamil Nadu, Pondy, Bihar, Jharkhand, Odisha, West Bengal, Sikkim, Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Tripura

---

## Future Improvements

- **More data** — hourly data or 5+ years would significantly improve accuracy
- **Feature engineering** — add temperature, holidays, weekday/weekend flags as additional input features
- **Transformer model** — attention mechanism may outperform LSTM for longer sequences
- **Hyperparameter tuning** — Optuna or Keras Tuner to find optimal units/dropout/lr per state
- **Ensemble** — combine LSTM with a simple statistical model (ARIMA) for better spike detection
- **Deployment** — wrap models in a FastAPI endpoint for real-time predictions

---

## Key Learnings / Bugs Fixed

1. **Loss explosion** — Original `relu` activation in LSTM caused MAE to reach billions for some states. Fixed with `tanh` + `clipnorm=1.0`
2. **Zero variance collapse** — With seq_length=30 on only 498 rows, the model collapsed to predicting the mean (loss=0.0000). Fixed by reducing to seq_length=7
3. **Shared scaler bug** — Original code used one MinMaxScaler for all states. States like Maharashtra (400 MW) and Sikkim (2 MW) need separate scalers or normalization destroys variance in small states
4. **Mac M1 slowdown** — Using `tf.keras.optimizers.legacy.Adam` instead of standard Adam gives 2–3x speedup on Apple Silicon

---

