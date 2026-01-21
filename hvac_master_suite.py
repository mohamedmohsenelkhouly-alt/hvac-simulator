import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HVAC Master Suite", layout="wide", initial_sidebar_state="expanded")

# --- GLOBAL CLASSES & UTILS ---

# 1. ASHRAE BUILDING PROFILES
ASHRAE_PROFILES = {
    "Residential": {
        "occ_density": 0.05, "equip_load": 5.0, "hours": "Residential", "u_value": 2.0, "fresh_air": 0.3
    },
    "Commercial (Office)": {
        "occ_density": 0.10, "equip_load": 15.0, "hours": "8to6", "u_value": 1.2, "fresh_air": 1.0
    },
    "Educational (School)": {
        "occ_density": 0.25, "equip_load": 10.0, "hours": "8to3", "u_value": 1.5, "fresh_air": 2.0
    }
}

class AshraeBuilding:
    def __init__(self, area, b_type):
        self.area = area
        self.type = b_type
        self.profile = ASHRAE_PROFILES[b_type]
        self.u_value = self.profile['u_value']
        
    def generate_schedule(self, length):
        hours = np.tile(np.arange(24), int(length/24) + 1)[:length]
        if self.profile['hours'] == "8to6":
            sched = np.where((hours >= 8) & (hours <= 18), 1.0, 0.1)
        elif self.profile['hours'] == "8to3":
            sched = np.where((hours >= 8) & (hours <= 15), 1.0, 0.05)
        else:
            sched = np.where((hours >= 18) | (hours <= 8), 1.0, 0.4)
        return sched

    def calculate_hourly_load(self, weather_df):
        sched = self.generate_schedule(len(weather_df))
        delta_t = (weather_df['Temperature'] - 24).clip(lower=0)
        
        # Physics: Q = U*A*dT + Internal + Vent
        q_cond = (self.u_value * self.area * delta_t) / 1000.0 
        people_load = (self.area * self.profile['occ_density']) * 100 
        equip_load = self.area * self.profile['equip_load']
        q_internal = ((people_load + equip_load) * sched) / 1000.0
        volume = self.area * 3.0
        q_vent = (0.33 * volume * self.profile['fresh_air'] * delta_t) / 1000.0
        
        total_load = q_cond + q_internal + q_vent
        weather_df['Load_kW'] = total_load
        return weather_df

@st.cache_data
def get_weather_data(years=1):
    hours = years * 8760
    dates = pd.date_range(start="2024-01-01", periods=hours, freq="h")
    x = np.linspace(0, years * 2 * np.pi, hours)
    temp = 22 + 12 * np.sin(x - np.pi/2) + 5 * np.sin(np.linspace(0, years*365*2*np.pi, hours)) + np.random.normal(0, 1, hours)
    return pd.DataFrame({"Date": dates, "Temperature": temp})

# --- NAVIGATION ---
st.sidebar.title("🔧 HVAC Master Suite")
app_mode = st.sidebar.radio("Select Module:", 
    ["1. Simple Fault Simulator", "2. ASHRAE Lifecycle Manager", "3. Campus Energy Predictor"])

# ==========================================
# MODULE 1: SIMPLE FAULT SIMULATOR
# ==========================================
if app_mode == "1. Simple Fault Simulator":
    st.title("❄️ Real-Time Degradation Simulator")
    st.markdown("Visualize how specific faults (clogged filters, fouling) affect power instantly.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Inject Faults")
        filter_clog = st.slider("Filter Clogging (%)", 0, 100, 0)
        coil_foul = st.slider("Condenser Fouling (%)", 0, 100, 0)
        sensor_bias = st.slider("Sensor Bias (°C)", -5.0, 5.0, 0.0)
        
        # Real-time Calcs
        base_power = 50.0 # kW
        deg_factor = (1 + filter_clog/200) * (1 + coil_foul/250)
        curr_power = base_power * deg_factor
        
        st.metric("Power Consumption", f"{curr_power:.1f} kW", delta=f"{curr_power-base_power:.1f} kW", delta_color="inverse")

    with col2:
        # Generate short-term data for viz
        df = pd.DataFrame({"Time": np.arange(0, 24)})
        df['Base'] = 50 + 10 * np.sin(df['Time']/24 * 2*np.pi)
        df['Degraded'] = df['Base'] * deg_factor
        df['Sensor'] = df['Base'] + sensor_bias # Abstract sensor rep
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Base'], name="Healthy", line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Degraded'], name="Faulty", line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODULE 2: ASHRAE LIFECYCLE MANAGER
# ==========================================
elif app_mode == "2. ASHRAE Lifecycle Manager":
    st.title("🏗️ ASHRAE Standard Lifecycle Manager")
    
    with st.expander("⚙️ Building Setup", expanded=True):
        col1, col2 = st.columns(2)
        b_type = col1.selectbox("Building Type", list(ASHRAE_PROFILES.keys()))
        b_area = col2.number_input("Floor Area (m²)", 1000, 50000, 2500)
    
    # Run Simulation
    weather = get_weather_data(20)
    b = AshraeBuilding(b_area, b_type)
    data = b.calculate_hourly_load(weather.copy())
    
    tab1, tab2, tab3 = st.tabs(["📉 Degradation", "🛠️ Retrofit", "🧠 Pareto Optimizer"])
    
    # --- DEGRADATION TAB ---
    with tab1:
        c1, c2 = st.columns(2)
        fl = c1.slider("Filter Interval (Months)", 1, 12, 3)
        cl = c2.slider("Coil Interval (Months)", 6, 24, 12)
        
        # Sim Logic
        d = data.copy()
        d['Month'] = (d['Date'].dt.year - d['Date'].dt.year.min()) * 12 + d['Date'].dt.month
        d['Year'] = d['Date'].dt.year - d['Date'].dt.year.min()
        
        f_pen = 1 + ((d['Month'] % fl)/fl) * 0.15
        c_pen = 1 + ((d['Month'] % cl)/cl) * 0.20
        comp_health = 1 - (d['Year'] * 0.01)
        
        d['Power'] = np.where(d['Load_kW']>0, (d['Load_kW']/3.2)*f_pen*c_pen/comp_health, 0)
        
        # Plot
        monthly = d.resample('M', on='Date')['Power'].sum().reset_index()
        fig = px.line(monthly, x='Date', y='Power', title="20-Year Energy Drift (ASHRAE Profile)")
        st.plotly_chart(fig, use_container_width=True)

    # --- RETROFIT TAB ---
    with tab2:
        col1, col2 = st.columns(2)
        r_year = col1.number_input("Retrofit Year", 5, 18, 10)
        r_cost = col2.number_input("Retrofit Cost ($)", 10000, 100000, 50000)
        
        if st.button("Run Retrofit Analysis"):
            # Baseline Cost
            base_cost = d['Power'].cumsum() * 0.15
            
            # Retrofit Logic
            d_r = d.copy()
            mask = d_r['Year'] >= r_year
            # Reset wear & boost eff
            d_r.loc[mask, 'Power'] = (d_r.loc[mask, 'Load_kW']/3.2) * (1/1.25) # 25% better
            
            retro_cost = d_r['Power'].cumsum() * 0.15
            # Add CAPEX
            idx = d_r[d_r['Year'] == r_year].index[0]
            retro_cost[idx:] += r_cost
            
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Scatter(x=d['Date'], y=base_cost, name="Do Nothing"))
            fig_roi.add_trace(go.Scatter(x=d['Date'], y=retro_cost, name="Retrofit"))
            st.plotly_chart(fig_roi, use_container_width=True)

    # --- PARETO TAB ---
    with tab3:
        if st.button("Run Pareto Optimization"):
            results = []
            for f in [1, 3, 6, 12]:
                for c in [6, 12, 24]:
                    # Quick Calc
                    n_maint = (240//f) + (240//c)
                    maint_cost = n_maint * 200 # $200 per visit
                    
                    # Energy approx (simplified for speed)
                    pen_avg = (1 + 0.07) * (1 + 0.1) # Avg penalty
                    e_cost = (d['Load_kW'].sum()/3.2) * pen_avg * 0.15
                    
                    results.append({"Setting": f"F{f}/C{c}", "Maint": maint_cost, "Energy": e_cost, "Total": maint_cost+e_cost})
            
            df_opt = pd.DataFrame(results)
            fig_p = px.scatter(df_opt, x="Maint", y="Energy", size="Total", color="Total", hover_name="Setting", title="Pareto Optimization")
            st.plotly_chart(fig_p, use_container_width=True)

# ==========================================
# MODULE 3: CAMPUS ENERGY PREDICTOR
# ==========================================
elif app_mode == "3. Campus Energy Predictor":
    st.title("🎓 Faculty Campus 20-Year Predictor")
    
    col1, col2 = st.columns(2)
    area = col1.number_input("Campus Area (m²)", 5000, 100000, 20000)
    density = col2.slider("Student Density (m²/student)", 1.0, 5.0, 2.5)
    
    if st.button("Generate Campus Forecast"):
        # Campus Logic (Heavy Vent + High Wear)
        dates = pd.date_range("2024-01-01", periods=20*8760, freq="h")
        years = dates.year - dates.year.min()
        
        # Load Growth (Degradation)
        base_load = (area * 0.15) # Base load kW
        drift = 1 + (years * 0.015) # 1.5% drift per year (High wear)
        
        

        energy = []
        for y in range(20):
            # Annual MWh with drift
            yr_load = base_load * drift[y*8760] * 2000 # 2000 equivalent hours
            energy.append(yr_load / 1000)
            
        df_camp = pd.DataFrame({"Year": range(2024, 2044), "MWh": energy})
        
        c1, c2 = st.columns(2)
        c1.metric("Year 1 MWh", f"{df_camp.iloc[0]['MWh']:,.0f}")
        c2.metric("Year 20 MWh", f"{df_camp.iloc[-1]['MWh']:,.0f}", delta=f"{(df_camp.iloc[-1]['MWh']/df_camp.iloc[0]['MWh']-1)*100:.1f}% Drift", delta_color="inverse")
        
        fig_c = px.bar(df_camp, x="Year", y="MWh", title="Forecasted Campus Energy Consumption")
        st.plotly_chart(fig_c, use_container_width=True)

