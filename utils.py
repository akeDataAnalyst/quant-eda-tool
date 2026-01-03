import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def load_data(uploaded_file, use_sample=False):
    if use_sample:
        df = pd.read_csv('synthetic_tick_data.csv', parse_dates=['timestamp'])
    else:
        df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    return df

def data_quality_report(df):
    st.subheader("Data Quality Report")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Duplicate Timestamps", df['timestamp'].duplicated().sum())
    col3.metric("Missing Prices", df['price'].isna().sum())
    col4.metric("Negative Prices", (df['price'] < 0).sum())
    
    st.write("### Detailed Issues")
    st.write(f"- Zero volume ticks: {(df['volume'] == 0).sum():,}")
    st.write(f"- Extreme outliers (>10× median price): {((df['price'] > 10 * df['price'].median()) & df['price'].notna()).sum():,}")

def clean_data(df):
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    df.loc[df['price'] < 0, 'price'] = np.nan
    df.loc[df['price'] > 10 * df['price'].median(), 'price'] = np.nan
    df['price'] = df['price'].ffill()
    df['volume'] = df['volume'].fillna(0)
    return df

def basic_eda(df):
    st.subheader("Basic Exploratory Analysis")
    
    # log_return is now pre-computed in app.py
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df['timestamp'], df['price'])
        ax.set_title('Price Time Series')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['log_return'].dropna(), bins=100, alpha=0.8, density=True)
        ax.set_title('Log Returns Distribution')
        ax.set_xlabel('Log Return')
        st.pyplot(fig)

def advanced_eda(df):
    st.subheader("Advanced Analysis")
    
    # Work on a copy with timestamp as index for time-based operations
    df_indexed = df.set_index('timestamp')
    
    # Ensure sorted
    df_indexed = df_indexed.sort_index()
    
    # Compute log_return if not already (safety — though we do it in app.py)
    if 'log_return' not in df_indexed.columns:
        df_indexed['log_return'] = np.log(df_indexed['price'] / df_indexed['price'].shift(1))
    
    # VWAP
    df_indexed['typical_price'] = df_indexed['price'] * df_indexed['volume']
    df_indexed['cum_vwap'] = df_indexed['typical_price'].cumsum() / df_indexed['volume'].cumsum()
    
    # Rolling volatility — use integer window as fallback if time-based fails
    try:
        # Try time-based first (preferred for irregular ticks)
        df_indexed['rolling_vol_15min'] = df_indexed['log_return'].rolling(window='15T').std()
    except:
        # Fallback: use 15-minute equivalent in number of ticks (robust)
        avg_ticks_per_min = len(df_indexed) / (24 * 60)  # ~13.9 ticks/min for 20k/day
        window_ticks = int(15 * avg_ticks_per_min)
        df_indexed['rolling_vol_15min'] = df_indexed['log_return'].rolling(window=max(1, window_ticks)).std()
    
    # Reset index for plotting consistency
    df_plot = df_indexed.reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # VWAP Plot
    ax1.plot(df_plot['timestamp'], df_plot['price'], label='Price', alpha=0.7, linewidth=0.8)
    ax1.plot(df_plot['timestamp'], df_plot['cum_vwap'], label='Cumulative VWAP', linewidth=2, color='orange')
    ax1.set_title('Price vs Cumulative VWAP')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rolling Volatility Plot
    ax2.plot(df_plot['timestamp'], df_plot['rolling_vol_15min'], color='red', linewidth=1)
    ax2.set_title('15-Minute Rolling Volatility (Std of Log Returns)')
    ax2.set_ylabel('Volatility')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

