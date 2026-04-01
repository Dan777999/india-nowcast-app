import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

# --- Page Config ---
st.set_page_config(page_title="India Nowcast Terminal", layout="wide")

# --- 1. DATA ENGINE (MOCKING 2026 LIVE FEEDS) ---
@st.cache_data
def get_india_macro_data():
    # In production, replace with: fetch_series(['RBI/GDP', 'MOSPI/IIP', 'SIAM/2W'])
    dates = pd.date_range(start='2025-04-01', periods=12, freq='MS')
    
    data = pd.DataFrame({
        'Month': dates,
        'Official_GDP_YoY': [7.2, 7.2, 7.2, 7.5, 7.5, 7.5, 7.8, 7.8, 7.8, np.nan, np.nan, np.nan],
        'IIP_Growth': [4.2, 4.5, 4.8, 5.1, 4.9, 5.5, 6.1, 6.3, 6.0, 6.4, 6.7, np.nan],
        'Electricity_Gen_BU': [140, 142, 145, 148, 147, 150, 155, 158, 157, 162, 165, 168],
        'Two_Wheeler_Sales': [1.5, 1.6, 1.55, 1.7, 1.8, 1.9, 2.1, 2.2, 2.15, 2.4, 2.5, 2.6], # Millions
        'Services_PMI': [58.2, 59.1, 58.5, 60.2, 61.5, 62.1, 61.8, 62.5, 63.2, 64.1, 65.0, 64.8]
    }).set_index('Month')
    return data

df = get_india_macro_data()

# --- 2. NOWCAST CALCULATOR ---
def run_nowcast_model(data):
    # We use IIP, Electricity, and 2W Sales to estimate the 'Factor'
    model_data = data[['IIP_Growth', 'Electricity_Gen_BU', 'Two_Wheeler_Sales', 'Services_PMI']].dropna(how='all')
    # Standardizing for DFM
    model_data_std = (model_data - model_data.mean()) / model_data.std()
    
    mod = DynamicFactor(model_data_std, k_factors=1, factor_order=1)
    res = mod.fit(disp=False)
    
    # Transform factor back to GDP scale (Approximate)
    factor = res.factors.filtered[0]
    nowcast_series = (factor * 0.5) + 7.5 # Centered around 2026 growth trend
    return nowcast_series

nowcast_estimates = run_nowcast_model(df)

# --- 3. UI LAYOUT ---
st.title("🇮🇳 India Economic Nowcast Terminal (v2026.1)")
st.markdown("Focus: **Actual Official Data** vs. **Real-Time High-Frequency Estimates**")

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
current_nowcast = nowcast_estimates[-1]
m1.metric("Current Nowcast (GDP %)", f"{current_nowcast:.2f}%", "+0.4% vs Prev Month")
m2.metric("Electricity Pulse", f"{df['Electricity_Gen_BU'].iloc[-1]} BU", "Strong")
m3.metric("Rural Demand (2W)", f"{df['Two_Wheeler_Sales'].iloc[-1]}M", "+26% YoY")
m4.metric("Services PMI", f"{df['Services_PMI'].iloc[-1]}", "Expansion")

st.divider()

# Charts
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("The Nowcast Curve")
    fig = go.Figure()
    # Actuals
    fig.add_trace(go.Scatter(x=df.index, y=df['Official_GDP_YoY'], name="Official GDP (Actual)", line=dict(color='white', width=4)))
    # Estimates
    fig.add_trace(go.Scatter(x=df.index, y=nowcast_estimates, name="Nowcast (Estimate)", line=dict(color='#00FFCC', dash='dot')))
    
    fig.update_layout(template="plotly_dark", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Advanced Add-ons")
    st.write("Electricity Generation (Industrial)")
    st.line_chart(df['Electricity_Gen_BU'])
    st.write("2-Wheeler Sales (Rural)")
    st.bar_chart(df['Two_Wheeler_Sales'])

# --- 4. DATA TABLE (ACTUALS VS ESTIMATES) ---
st.subheader("The 'Ragged Edge' Analyst Table")
st.markdown("Compare raw government data releases against the model's imputed estimates.")

# Creating the Comparison Table
table_df = df.copy()
table_df['Model_Nowcast_Estimate'] = nowcast_estimates
# Formatting for readability
table_df.index = table_df.index.strftime('%b-%Y')
st.dataframe(
    table_df.style.highlight_null(color="#421a1a")
    .format(precision=2),
    use_container_width=True
)

st.caption("Note: Red cells indicate 'Pending' official releases where the Nowcast is currently providing the signal.")
