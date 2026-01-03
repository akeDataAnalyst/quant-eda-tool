import streamlit as st
import pandas as pd
import numpy as np  
from utils import load_data, data_quality_report, clean_data, basic_eda, advanced_eda

# Page config
st.set_page_config(page_title="Quant EDA Tool", layout="wide")
st.title("🧮 Internal Quant Tool for Exploratory Data Analysis")
st.markdown("""
This internal utility supports quantitative researchers by automating:
- Data quality validation
- One-click cleaning
- Basic and advanced exploratory analysis

Upload your own tick dataset or use the built-in synthetic BTCUSDT example.
""")

# Sidebar
st.sidebar.header("Data Source")
use_sample = st.sidebar.checkbox("Use synthetic BTCUSDT tick data (demo)", value=True)

if use_sample:
    df_raw = load_data(None, use_sample=True)
    st.sidebar.success(f"Loaded synthetic_tick_data.csv ({len(df_raw):,} rows)")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (required columns: timestamp, price, volume)",
        type=['csv']
    )
    if uploaded_file is None:
        st.info("👆 Please upload a CSV file or check 'Use synthetic data' to continue.")
        st.stop()
    df_raw = load_data(uploaded_file)

# Raw data preview
with st.expander("🔍 View Raw Data Preview", expanded=False):
    st.dataframe(df_raw.head(1000))
    st.write(f"Shape: {df_raw.shape}")

# Data Quality Report
data_quality_report(df_raw)

# Cleaning Section
st.subheader("🧹 Data Cleaning")
if st.button("Clean Dataset (dedupe, fix negatives/outliers, forward-fill price)"):
    with st.spinner("Cleaning data..."):
        df_cleaned = clean_data(df_raw)
    st.success(f"✅ Cleaning complete: {len(df_cleaned):,} rows retained")
    
    # Save to session state
    st.session_state.df_clean = df_cleaned
    st.session_state.cleaned = True

# Check if we have cleaned data
if 'df_clean' not in st.session_state:
    st.info("👆 Click the 'Clean Dataset' button above to enable analysis tabs.")
    st.stop()

# Retrieve cleaned data safely
df_clean = st.session_state['df_clean'].copy()

# CRITICAL: Create log_return here so it's always available for both tabs
df_clean['log_return'] = np.log(df_clean['price'] / df_clean['price'].shift(1))

# Now set timestamp as index
df_clean = df_clean.set_index('timestamp')

st.success(f"📊 Analysis ready — using cleaned dataset with {len(df_clean):,} rows")

# Analysis Tabs
tab1, tab2 = st.tabs(["Basic EDA", "Advanced Analysis"])

with tab1:
    basic_eda(df_clean.copy().reset_index())  # Functions expect 'timestamp' column

with tab2:
    advanced_eda(df_clean.copy().reset_index())

# Footer
st.caption("Built with Python • Pandas • NumPy • Streamlit | Clean, readable code for quant research support")
st.caption("Developed by Aklilu Abera | Quant EDA Tool for Research Support")


