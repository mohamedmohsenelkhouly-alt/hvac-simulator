import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HVAC Degradation Simulator", layout="wide")

st.title("❄️ HVAC System Degradation Simulator")
st.markdown("""
This tool simulates how different physical faults affect HVAC power consumption and efficiency.
Adjust the **Degradation Parameters** on the left to see the impact.
""")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("1. Simulation Settings")
duration = st.sidebar.slider("Simulation Duration (Hours)", 24, 168, 48) # Default 48 hours
base_load = st.sidebar.number_input("Base Cooling Load (kW)", value=50.0)

st.sidebar.header("2. Inject Faults")

# Fault 1: Filter Clogging
st.sidebar.subheader("Filter Clogging")
filter_clog_pct = st.sidebar.slider("Filter Blockage (%)", 0, 100, 0)
# Logic: Blockage increases fan resistance. 
# Simplified physics: Power increases cubically with flow/resistance in some models, 
# but linearly for simple resistance approx here.
filter_factor = 1 + (filter_clog_pct / 100.0) * 0.5  # Max 50% extra power at 100% clog

# Fault 2: Condenser Fouling
st.sidebar.subheader("Condenser Coil Fouling")
fouling_pct = st.sidebar.slider("Fouling Level (%)", 0, 100, 0)
# Logic: Fouling reduces heat transfer, raising discharge pressure and compressor work.
fouling_factor = 1 + (fouling_pct / 100.0) * 0.4 # Max 40% extra power

# Fault 3: Sensor Drift
st.sidebar.subheader("Temp Sensor Drift")
sensor_bias = st.sidebar.slider("Sensor Bias (°C)", -5.0, 5.0, 0.0)

# --- DATA GENERATION ENGINE ---
def generate_data(hours, load, filter_f, fouling_f, bias):
    # Create time index
    time_index = pd.date_range(start="2024-01-01", periods=hours, freq="H")
    
    # 1. Generate Healthy Data (Synthetic)
    # Simulate a daily cycle (sin wave) + random noise
    x = np.linspace(0, hours/24 * 2 * np.pi, hours)
    ambient_temp = 25 + 10 * np.sin(x - 3) + np.random.normal(0, 0.5, hours) # Day/Night cycle
    
    # Healthy Power is proportional to Ambient Temp (simplified)
    healthy_power = (load * 0.5) + (ambient_temp * 0.8) 
    
    # 2. Apply Degradations
    # Combined degradation factor (Multiplicative)
    total_degradation = filter_f * fouling_f
    
    degraded_power = healthy_power * total_degradation
    
    # 3. Apply Sensor Fault (Drift)
    # This affects the "Read" temperature, not necessarily power in this simple viz,
    # but shows data quality issues.
    sensor_reading = ambient_temp + bias
    
    return pd.DataFrame({
        "Time": time_index,
        "Ambient_Temp_Actual": ambient_temp,
        "Ambient_Temp_Sensor": sensor_reading,
        "Power_Healthy_kW": healthy_power,
        "Power_Degraded_kW": degraded_power
    })

# Generate the data based on user inputs
df = generate_data(duration, base_load, filter_factor, fouling_factor, sensor_bias)

# --- DASHBOARD LAYOUT ---

# Top Metrics
col1, col2, col3 = st.columns(3)
total_healthy = df['Power_Healthy_kW'].sum()
total_degraded = df['Power_Degraded_kW'].sum()
extra_cost = (total_degraded - total_healthy) * 0.15 # Assuming $0.15 per kWh

col1.metric("Healthy Energy Consumed", f"{total_healthy:,.0f} kWh")
col2.metric("Degraded Energy Consumed", f"{total_degraded:,.0f} kWh", delta_color="inverse")
col3.metric("Estimated Wasted Cost", f"${extra_cost:.2f}", delta=f"+{(total_degraded/total_healthy - 1)*100:.1f}%")

st.markdown("---")

# Chart 1: Power Consumption Comparison
st.subheader("⚡ Power Consumption: Healthy vs. Degraded")
fig_power = go.Figure()
fig_power.add_trace(go.Scatter(x=df['Time'], y=df['Power_Healthy_kW'], 
                         mode='lines', name='Healthy System', line=dict(color='green')))
fig_power.add_trace(go.Scatter(x=df['Time'], y=df['Power_Degraded_kW'], 
                         mode='lines', name='Degraded System', line=dict(color='red', dash='dash')))
fig_power.update_layout(xaxis_title="Time", yaxis_title="Power (kW)", height=400)
st.plotly_chart(fig_power, use_container_width=True)

# Chart 2: Sensor Analysis
st.subheader("🌡️ Sensor Diagnostics")
st.write("If you apply a **Sensor Bias**, observe how the reported temperature deviates from reality.")
fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(x=df['Time'], y=df['Ambient_Temp_Actual'], 
                         mode='lines', name='Actual Temp', line=dict(color='blue')))
fig_temp.add_trace(go.Scatter(x=df['Time'], y=df['Ambient_Temp_Sensor'], 
                         mode='lines', name='Sensor Reading', line=dict(color='orange')))
st.plotly_chart(fig_temp, use_container_width=True)

# Data Table
with st.expander("View Raw Simulation Data"):
    st.dataframe(df)

