import streamlit as st
import pandas as pd
import os
from PIL import Image

st.set_page_config(
    page_title="Energy Consumption Forecaster",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecasting")
st.markdown("### LSTM-based forecasting across Indian states")
st.markdown("---")

# Get all regions from plots folder
plots_dir = "plots"
prediction_plots = [f for f in os.listdir(plots_dir) 
                   if f.endswith("_predictions.png")]
regions = sorted([f.replace("_predictions.png", "") 
                 for f in prediction_plots])

# Sidebar
st.sidebar.title("Select Region")
selected_region = st.sidebar.selectbox("Choose a state:", regions)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📈 {selected_region} — Actual vs Predicted")
    pred_img = Image.open(f"plots/{selected_region}_predictions.png")
    st.image(pred_img, use_column_width=True)

with col2:
    st.subheader(f"📉 {selected_region} — Training Loss")
    loss_img = Image.open(f"plots/{selected_region}_loss.png")
    st.image(loss_img, use_column_width=True)

st.markdown("---")
st.subheader("🌍 All Regions Comparison")
comparison_img = Image.open("plots/india_all_states_comparison.png")
st.image(comparison_img, use_column_width=True)

# Show raw data
if st.sidebar.checkbox("Show cleaned data"):
    cleaned_file = f"cleaned_data/{selected_region}_cleaned.csv"
    if os.path.exists(cleaned_file):
        df = pd.read_csv(cleaned_file)
        st.subheader(f"📊 {selected_region} — Cleaned Dataset")
        st.dataframe(df.head(100))
        st.write(f"Total rows: {len(df):,}")