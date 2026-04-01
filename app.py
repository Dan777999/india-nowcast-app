import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dbnomics import fetch_series
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

st.set_page_config(page_title="India Nowcast Terminal", layout="wide")

## --- DATA FETCHING ---
@st.cache_data
def load_india_data():
    # Example series IDs for India (IDs may vary by DBnomics provider)
    # Using RBI/MOSPI through DBnomics
    series = {
        'GDP': 'RBI/QAB/QAB_NGDP_201112', # Quarterly GDP
        'IIP': 'RBI/MAB/MAB_IIP_201112',  # Monthly Ind. Production
        'CPI': 'RBI/MAB/MAB_CPI_2012',    # Monthly Inflation
    }
    # For a real app, you'd fetch these using:
    # df = fetch_series(series_id)
    
    # MOCK DATA for demonstration (Mimicking India's current trends)
    dates = pd.date_range(start='2022-01-01', periods=24, freq='MS')
    data = pd.DataFrame({
        'IIP': np.random.normal(5, 2, 24).cumsum(),
        'Services_PMI': np.random.normal(55, 3, 24),
        'GST_Coll': np.random.normal(1.6, 0.1, 24) # in Trillion INR
    }, index=dates)
    return data

df_monthly = load_india_data()

## --- NOWCAST ENGINE ---
def run_nowcast(data):
    # Dynamic Factor Model: Extracts the "Common Factor" from multiple HFIs
    # This represents the underlying 'pulse' of the Indian economy
    model = DynamicFactor(data, k_factors=1, factor_order=1)
    results = model.fit(disp=False)
    # The 'factor' is our nowcast proxy
    nowcast_val = results.factors.filtered[0][-1]
    return nowcast_val, results

## --- UI LAYOUT ---
st.title("🇮🇳 India Economic Nowcast Terminal")
st.markdown("Estimating current activity using high-frequency monthly indicators.")

col1, col2, col3 = st.columns(3)
nowcast_score, res = run_nowcast(df_monthly)

with col1:
    st.metric("Estimated GDP Momentum", f"{nowcast_score:.2f}%", "+0.4%")
with col2:
    st.metric("Latest GST Collection", "₹1.78T", "11% YoY")
with col3:
    st.metric("Services PMI", "61.2", "Expanding")

## --- VISUALIZATION ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly['IIP'], name="Ind. Production"))
fig.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly['Services_PMI'], name="Services PMI", yaxis="y2"))

fig.update_layout(
    title="High-Frequency Indicator Pulse",
    yaxis2=dict(overlaying='y', side='right'),
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

st.info("Note: This model uses a Dynamic Factor Model (DFM) to synthesize monthly noise into a signal.")
