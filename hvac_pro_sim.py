import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURATION ---
st.set_page_config(page_title="HVAC Pro: 20-Year Degradation Sim", layout="wide")

st.title("🏭 HVAC Lifecycle Simulator (20 Years)")
st.markdown("""
This tool simulates long-term HVAC degradation based on **operating hours** and **weather conditions**.
It models the accumulation of wear (fouling, clogging, aging) and the impact of maintenance schedules.
""")

# --- SIDEBAR: SYSTEM INPUTS ---
with st.sidebar:
    st.header("1. System Setup")
    base_capacity_kw = st.number_input("System Capacity (kW)", value=100.0)
    base_cop = st.number_input("Rated COP", value=3.5)
    
    st.header("2. Maintenance Schedule")
    filter_life = st.number_input("Filter Change Interval (Months)", value=3, min_value=1)
    coil_cleaning = st.number_input("Coil Cleaning Interval (Months)", value=12, min_value=1)
    
    st.header("3. Degradation Rates")
    # Filters clog fast, Coils foul medium, Compressors wear slow
    filter_deg_rate = st.slider("Filter Clogging Rate", 0.1, 2.0, 1.0, help="Impact on fan power per month")
    coil_deg_rate = st.slider("Coil Fouling Rate", 0.1, 2.0, 1.0, help="Impact on heat transfer per month")
    comp_wear_rate = st.slider("Compressor Aging (%/Year)", 0.0, 5.0, 1.0, help="Permanent efficiency loss per year")

    st.header("4. Weather Data")
    data_source = st.radio("Weather Source", ["Generate Synthetic (20 Years)", "Upload CSV"])
    
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV (cols: Date, Temperature)", type="csv")

# --- SIMULATION ENGINE ---

def generate_weather_data(years=20):
    """Generates 20 years of hourly temperature data with seasonal variations."""
    hours = years * 8760
    dates = pd.date_range(start="2024-01-01", periods=hours, freq="h")
    
    # Yearly Cycle (Seasonality)
    x = np.linspace(0, years * 2 * np.pi, hours)
    seasonal_temp = 15 * np.sin(x - np.pi/2) + 20 # 5°C to 35°C swing
    
    # Daily Cycle (Day/Night)
    day_cycle = 5 * np.sin(np.linspace(0, years * 365 * 2 * np.pi, hours) - 3)
    
    # Random Weather Noise
    noise = np.random.normal(0, 2, hours)
    
    temp = seasonal_temp + day_cycle + noise
    return pd.DataFrame({"Date": dates, "Temperature": temp})

@st.cache_data
def run_simulation(df, cap, cop, f_life, c_life, f_rate, c_rate, comp_rate):
    # 1. Calculate Base Load (Weather Dependent)
    # Assumption: Cooling starts when Temp > 18°C
    df['Cooling_Load_kW'] = (df['Temperature'] - 18).clip(lower=0) * (cap / 15.0)
    
    # Identify Operating Hours (System is ON if Load > 0)
    df['System_ON'] = (df['Cooling_Load_kW'] > 0).astype(int)
    
    # 2. Initialize Degradation Vectors
    n = len(df)
    
    # --- A. Filter Clogging (Sawtooth Pattern) ---
    # Resets every 'f_life' months
    # We calculate "Months since last change"
    df['Month_Index'] = (df['Date'].dt.year - df['Date'].dt.year.min()) * 12 + df['Date'].dt.month
    df['Filter_Cycle'] = df['Month_Index'] % f_life
    # Factor: Starts at 1.0, grows based on rate and how far into the cycle we are
    df['Filter_Penalty'] = 1 + (df['Filter_Cycle'] * 0.02 * f_rate) 
    
    # --- B. Coil Fouling (Sawtooth Pattern) ---
    # Resets every 'c_life' months
    df['Coil_Cycle'] = df['Month_Index'] % c_life
    # Factor: Increases head pressure, reducing efficiency
    df['Coil_Penalty'] = 1 + (df['Coil_Cycle'] * 0.01 * c_rate)
    
    # --- C. Compressor Wear (Linear/Exponential, Permanent) ---
    # Does not reset. Depends on cumulative years.
    df['Year_Index'] = df['Date'].dt.year - df['Date'].dt.year.min()
    # Efficiency drops: 100% -> 99% -> 98% ...
    df['Comp_Efficiency_Factor'] = 1 - (df['Year_Index'] * (comp_rate / 100.0))
    
    # 3. Calculate Final Power Consumption
    # Power = (Load / COP) * Penalties / Efficiency
    
    # Base Power needed without faults
    df['Power_Base_kW'] = np.where(df['Cooling_Load_kW'] > 0, 
                                   df['Cooling_Load_kW'] / cop, 
                                   0)
    
    # Actual Power with faults
    # Filter adds resistance (Fan power) -> Multiplier
    # Coil adds lift (Comp power) -> Multiplier
    # Comp wear reduces capacity -> Divisor
    
    total_penalty = df['Filter_Penalty'] * df['Coil_Penalty']
    
    df['Power_Actual_kW'] = (df['Power_Base_kW'] * total_penalty) / df['Comp_Efficiency_Factor']
    
    return df

# --- MAIN EXECUTION ---

# 1. Load Data
if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        # Resample to hourly if needed, for now assume hourly
        st.success("Custom weather file loaded!")
    except:
        st.error("Error reading CSV. Ensure columns are 'Date' and 'Temperature'")
        st.stop()
else:
    raw_df = generate_weather_data(years=20)

# 2. Run Sim
with st.spinner('Simulating 20 years of physics...'):
    sim_df = run_simulation(raw_df, base_capacity_kw, base_cop, 
                            filter_life, coil_cleaning, 
                            filter_deg_rate, coil_deg_rate, comp_wear_rate)

# 3. Aggregation (for performance plotting)
# Group by Month to show the 20-year trend clearly
monthly_df = sim_df.groupby(pd.Grouper(key='Date', freq='M')).agg({
    'Power_Base_kW': 'sum',
    'Power_Actual_kW': 'sum',
    'Temperature': 'mean',
    'Filter_Penalty': 'max',
    'Coil_Penalty': 'max',
    'Comp_Efficiency_Factor': 'min'
}).reset_index()

monthly_df['Wasted_Energy_kWh'] = monthly_df['Power_Actual_kW'] - monthly_df['Power_Base_kW']
monthly_df['Cost_Waste'] = monthly_df['Wasted_Energy_kWh'] * 0.15 # $0.15/kWh

# --- VISUALIZATIONS ---

# Tab Layout
tab1, tab2, tab3 = st.tabs(["📊 20-Year Overview", "🔍 Component Diagnostics", "💸 Financial Impact"])

with tab1:
    st.subheader("Total System Degradation Over 20 Years")
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=monthly_df['Date'], y=monthly_df['Power_Base_kW'], 
                                  mode='lines', name='Ideal Energy (New System)', line=dict(color='green', width=1)))
    fig_main.add_trace(go.Scatter(x=monthly_df['Date'], y=monthly_df['Power_Actual_kW'], 
                                  mode='lines', name='Actual Energy (Degraded)', line=dict(color='red', width=1)))
    fig_main.update_layout(yaxis_title="Monthly Energy (kWh)", hovermode="x unified")
    st.plotly_chart(fig_main, use_container_width=True)
    
    st.info("Notice the 'Sawtooth' pattern? That is your maintenance schedule (cleaning coils/filters) resetting efficiency temporarily, while the gap slowly widens due to permanent compressor aging.")

with tab2:
    st.subheader("Deep Dive: Physics Factors")
    
    col1, col2 = st.columns(2)
    
    # Filter & Coil Cycles
    fig_maint = go.Figure()
    fig_maint.add_trace(go.Scatter(x=monthly_df['Date'], y=monthly_df['Filter_Penalty'], name="Filter Resistance", line=dict(color='orange')))
    fig_maint.add_trace(go.Scatter(x=monthly_df['Date'], y=monthly_df['Coil_Penalty'], name="Coil Fouling", line=dict(color='blue')))
    fig_maint.update_layout(title="Maintenance Cycles (Recoverable Degradation)", yaxis_title="Penalty Factor (1.0 = Clean)")
    col1.plotly_chart(fig_maint, use_container_width=True)
    
    # Compressor Wear
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=monthly_df['Date'], y=monthly_df['Comp_Efficiency_Factor']*100, 
                                  name="Compressor Health", line=dict(color='black', dash='dot')))
    fig_comp.update_layout(title="Compressor Aging (Permanent Degradation)", yaxis_title="Efficiency %")
    col2.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    st.subheader("Cumulative Financial Loss")
    total_waste = monthly_df['Cost_Waste'].sum()
    st.metric("Total Wasted Money over 20 Years", f"${total_waste:,.2f}")
    
    fig_cost = px.bar(monthly_df, x='Date', y='Cost_Waste', title="Monthly Cost of Inefficiency ($)")
    fig_cost.update_traces(marker_color='red')
    st.plotly_chart(fig_cost, use_container_width=True)

# Data Export
st.sidebar.markdown("---")
csv = sim_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download Full Simulation Data (CSV)", csv, "hvac_20yr_sim.csv", "text/csv")
