import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="ASHRAE HVAC Simulator Pro", layout="wide", initial_sidebar_state="expanded")

# --- ASHRAE STANDARD DEFAULTS (Approximations based on Std 90.1 & Fundamentals) ---
ASHRAE_PROFILES = {
    "Residential": {
        "occ_density": 0.05,  # People per m2 (Low)
        "equip_load": 5.0,    # W/m2 (Lighting + Plug)
        "hours": "Residential", # Custom logic
        "u_value": 2.0,       # Older code compliant
        "fresh_air": 0.3      # ACH (Air Changes per Hour) - Infiltration dominated
    },
    "Commercial (Office)": {
        "occ_density": 0.10,  # ~10m2 per person
        "equip_load": 15.0,   # W/m2 (Computers + Lights)
        "hours": "8to6",      # Standard Work Day
        "u_value": 1.2,       # Glass curtain wall / modern insulation
        "fresh_air": 1.0      # ASHRAE 62.1 Ventilation req
    },
    "Educational (School)": {
        "occ_density": 0.25,  # High density (Classrooms)
        "equip_load": 10.0,   # W/m2
        "hours": "8to3",      # School Day
        "u_value": 1.5,       # Brick/Block construction
        "fresh_air": 2.0      # High ventilation requirement (Classrooms)
    }
}

# --- CLASS: BUILDING PHYSICS ENGINE ---
class AshraeBuilding:
    def __init__(self, area, b_type, custom_u=None):
        self.area = area
        self.type = b_type
        self.profile = ASHRAE_PROFILES[b_type]
        # Allow override of U-value if user desires, else use Standard
        self.u_value = custom_u if custom_u else self.profile['u_value']
        
    def generate_schedule(self, length):
        # Generate hourly schedule (0.0 to 1.0) based on building type
        hours = np.tile(np.arange(24), int(length/24) + 1)[:length]
        
        if self.profile['hours'] == "8to6":
            # Office: Peak 8am-6pm, Low otherwise
            sched = np.where((hours >= 8) & (hours <= 18), 1.0, 0.1)
            # Weekend shutdown (Simplified: Every 6th/7th day is low)
            # For simplicity in this demo, we assume 7-day op or average it out
        
        elif self.profile['hours'] == "8to3":
            # School: Peak 8am-3pm
            sched = np.where((hours >= 8) & (hours <= 15), 1.0, 0.05)
            
        else: # Residential
            # People home at night (6pm - 8am)
            sched = np.where((hours >= 18) | (hours <= 8), 1.0, 0.4)
            
        return sched

    def calculate_hourly_load(self, weather_df):
        # Q_total = Q_cond + Q_internal + Q_ventilation
        
        # 1. Schedule
        sched = self.generate_schedule(len(weather_df))
        
        # 2. Conduction (Q = U*A*dT)
        # Cooling Setpoint: 24°C (ASHRAE Comfort Zone Summer)
        delta_t = (weather_df['Temperature'] - 24).clip(lower=0)
        q_cond = (self.u_value * self.area * delta_t) / 1000.0 # kW
        
        # 3. Internal Gains (People + Equip) * Schedule
        # Heat gain per person ~ 100W sensible
        people_load = (self.area * self.profile['occ_density']) * 100 
        equip_load = self.area * self.profile['equip_load']
        q_internal = ((people_load + equip_load) * sched) / 1000.0 # kW
        
        # 4. Ventilation / Infiltration (Q = 1.08 * CFM * dT ... simplified metric)
        # Using simple metric: 0.33 * Volume * ACH * dT (Volume approx Area * 3m height)
        volume = self.area * 3.0
        q_vent = (0.33 * volume * self.profile['fresh_air'] * delta_t) / 1000.0 # kW
        
        total_load = q_cond + q_internal + q_vent
        
        weather_df['Load_kW'] = total_load
        weather_df['Schedule'] = sched
        return weather_df

# --- HELPER FUNCTIONS ---
@st.cache_data
def get_weather_data(years=20):
    hours = years * 8760
    dates = pd.date_range(start="2024-01-01", periods=hours, freq="h")
    # Base Temp + Seasonality + Diurnal + Random
    x = np.linspace(0, years * 2 * np.pi, hours)
    temp = 22 + 12 * np.sin(x - np.pi/2) # Seasonal
    day = 5 * np.sin(np.linspace(0, years * 365 * 2 * np.pi, hours)) # Daily
    noise = np.random.normal(0, 1, hours)
    return pd.DataFrame({"Date": dates, "Temperature": temp + day + noise})

def simulate_hvac(df, params):
    d = df.copy()
    
    # Unpack Maintenance Params
    f_life = params['filter_life']
    c_life = params['coil_life']
    # Retrofit Params
    r_year = params.get('retro_year', 99)
    r_gain = params.get('retro_gain', 0)
    
    # 1. Time Indexing
    d['Month_Abs'] = (d['Date'].dt.year - d['Date'].dt.year.min()) * 12 + d['Date'].dt.month
    d['Year_Abs'] = d['Date'].dt.year - d['Date'].dt.year.min()
    
    # 2. Degradation Factors (Physics-based penalties)
    # Filter: Approaches 20% flow restriction at end of life
    d['F_Pen'] = 1 + ((d['Month_Abs'] % f_life) / f_life) * 0.15 
    
    # Coil: Approaches 25% heat transfer loss at end of life
    d['C_Pen'] = 1 + ((d['Month_Abs'] % c_life) / c_life) * 0.20
    
    # Compressor: 1% efficiency loss per year (Accumulative)
    # Handle Retrofit Reset
    is_retro = d['Year_Abs'] >= r_year
    age = np.where(is_retro, d['Year_Abs'] - r_year, d['Year_Abs'])
    eff_drop = age * 0.01 
    d['Comp_Eff'] = 1.0 - eff_drop
    
    # 3. Power Calculation
    # Power = (Load / COP_Design) * (Penalties) / (Eff_Factor)
    cop_base = 3.2 # Standard Chiller/DX unit
    
    retro_boost = np.where(is_retro, 1 + r_gain, 1.0)
    
    # Only run HVAC when Load > 0
    d['Power_kW'] = np.where(d['Load_kW'] > 0, 
                             (d['Load_kW'] / cop_base) * d['F_Pen'] * d['C_Pen'] / (d['Comp_Eff'] * retro_boost), 
                             0)
    
    return d

# --- APP UI START ---
st.title("🏗️ ASHRAE HVAC Lifecycle Manager")
st.markdown("A degradation and retrofit simulator compliant with **ASHRAE Fundamentals & Std 90.1** building profiles.")

# SIDEBAR
with st.sidebar:
    st.header("1. Building Configuration")
    b_type = st.selectbox("Building Type", list(ASHRAE_PROFILES.keys()))
    b_area = st.number_input("Total Floor Area (m²)", 500, 50000, 2500)
    
    st.info(f"**ASHRAE Profile Loaded:**\n"
            f"- Occupancy: {ASHRAE_PROFILES[b_type]['occ_density']} p/m²\n"
            f"- Loads: {ASHRAE_PROFILES[b_type]['equip_load']} W/m²\n"
            f"- Vent: {ASHRAE_PROFILES[b_type]['fresh_air']} ACH")
    
    st.header("2. Simulation Settings")
    sim_years = 20

# INITIALIZE DATA
weather = get_weather_data(sim_years)
building = AshraeBuilding(b_area, b_type)
load_data = building.calculate_hourly_load(weather.copy())

# TABS
tab1, tab2, tab3 = st.tabs(["📉 Degradation & Maintenance", "🛠️ Retrofit ROI", "🧠 Pareto Optimizer"])

# --- TAB 1: DEGRADATION ---
with tab1:
    st.subheader("System Performance Over Time")
    
    col1, col2 = st.columns(2)
    with col1:
        fl = st.slider("Filter Change Interval (Months)", 1, 12, 3)
    with col2:
        cl = st.slider("Coil Cleaning Interval (Months)", 6, 36, 12)
        
    # Run Baseline Sim
    base_params = {'filter_life': fl, 'coil_life': cl}
    res = simulate_hvac(load_data, base_params)
    
    # Resample for Plotting (Monthly Sum)
    monthly = res.set_index('Date').resample('M').agg({'Power_kW': 'sum', 'Load_kW': 'sum'}).reset_index()
    monthly['Efficiency_Index'] = monthly['Load_kW'] / monthly['Power_kW'] # Effective COP
    
    fig = px.line(monthly, x='Date', y='Power_kW', title="Monthly Energy Consumption (kWh)")
    fig.add_trace(go.Scatter(x=monthly['Date'], y=monthly['Power_kW'].rolling(12).mean(), 
                             name='12-Month Moving Avg', line=dict(color='red', width=3)))
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Engineering Logic"):
        st.write("""
        - **Filter Degradation:** Modeled as flow resistance increasing fan power. Resets every `Filter Interval`.
        - **Coil Fouling:** Modeled as reduced heat transfer effectiveness (UA), increasing compressor lift. Resets every `Coil Interval`.
        - **Compressor Wear:** Modeled as standard isentropic efficiency loss (1% per year), non-recoverable without retrofit.
        """)

# --- TAB 2: RETROFIT ROI ---
with tab2:
    st.subheader("Retrofit Decision Support")
    
    c1, c2, c3 = st.columns(3)
    r_year = c1.number_input("Retrofit at Year", 5, 18, 10)
    r_eff = c2.number_input("Efficiency Improvement (%)", 10, 50, 25) / 100.0
    r_capex = c3.number_input("Retrofit Cost ($)", 10000, 500000, 50000)
    
    if st.button("Calculate ROI"):
        # 1. Baseline (No Retrofit)
        df_base = simulate_hvac(load_data, {'filter_life': 3, 'coil_life': 12})
        cost_base = df_base['Power_kW'].cumsum() * 0.15 # $0.15/kWh
        
        # 2. Retrofit Scenario
        p_retro = {'filter_life': 3, 'coil_life': 12, 'retro_year': r_year, 'retro_gain': r_eff}
        df_retro = simulate_hvac(load_data, p_retro)
        
        # Add CAPEX spike
        cost_retro_series = df_retro['Power_kW'].cumsum() * 0.15
        idx = df_retro[df_retro['Year_Abs'] == r_year].index[0]
        cost_retro_series[idx:] += r_capex
        
        # Plot
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(x=df_base['Date'], y=cost_base, name="Existing System", line=dict(color='grey')))
        fig_roi.add_trace(go.Scatter(x=df_retro['Date'], y=cost_retro_series, name="With Retrofit", line=dict(color='green')))
        
        st.plotly_chart(fig_roi, use_container_width=True)
        
        net_savings = cost_base.iloc[-1] - cost_retro_series.iloc[-1]
        st.metric("Net Lifetime Savings", f"${net_savings:,.2f}", delta_color="normal" if net_savings > 0 else "inverse")

# --- TAB 3: PARETO OPTIMIZER ---
with tab3:
    st.subheader("Multi-Objective Maintenance Optimization")
    st.markdown("Finding the ASHRAE-compliant 'Sweet Spot' between **Maintenance Spend** and **Energy Waste**.")
    
    maint_cost = st.number_input("Cost per Maintenance Visit ($)", 100, 1000, 200)
    
    if st.button("Run Optimization"):
        # Discrete Search Space
        filters = [1, 2, 3, 6, 12]
        coils = [6, 12, 18, 24]
        
        results = []
        
        with st.spinner("Simulating Scenarios..."):
            for f in filters:
                for c in coils:
                    # Run 20 year sim
                    s = simulate_hvac(load_data, {'filter_life': f, 'coil_life': c})
                    
                    # Energy Cost
                    energy_bill = s['Power_kW'].sum() * 0.15
                    
                    # Maint Cost (Count events)
                    n_f = (20*12) // f
                    n_c = (20*12) // c
                    maint_bill = (n_f + n_c) * maint_cost
                    
                    results.append({
                        "Setting": f"F:{f}m / C:{c}m",
                        "Energy_Cost": energy_bill,
                        "Maint_Cost": maint_bill,
                        "Total": energy_bill + maint_bill
                    })
        
        df_opt = pd.DataFrame(results)
        
        # Plot
        fig_par = px.scatter(df_opt, x="Maint_Cost", y="Energy_Cost", color="Total", 
                             hover_data=["Setting"], size="Total",
                             title="Pareto Front: Maintenance vs Energy")
        st.plotly_chart(fig_par, use_container_width=True)
        
        best = df_opt.loc[df_opt['Total'].idxmin()]
        st.success(f"Recommended Strategy: {best['Setting']} (Lowest TCO: ${best['Total']:,.0f})")

