import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
import os

# To run on port 8074: python -m streamlit run FinalUpdate.py --server.port 8074

st.set_page_config(layout="wide", page_title="Network AI Decision Support")
st.title("Network Traffic AI Decision Support System")
st.markdown("### LGBM Forecast + DQN Action")

MODEL_PATH = "modelS.txt"

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("Step 1: Upload Files")
    u_raw_data = st.file_uploader("Upload Anonymized_Dataset.csv", type="csv")

    # Dynamic visual indicators for file persistence
    if os.path.exists(MODEL_PATH):
        st.success(f"'{MODEL_PATH}' found on disk. System will skip training.")
    else:
        st.warning(f"'{MODEL_PATH}' not found. System will train a new one upon upload.")

    st.divider()
    st.header("Step 2: Strategy Settings")
    sla_p = st.slider("SLA / Performance Priority", 0.0, 2.0, 1.0, 0.1)
    energy_p = st.slider("Energy Saving Priority", 0.0, 2.0, 1.0, 0.1)


#CACHED STEP 1: LOAD & VECTORIZE DATA PROCESSING
@st.cache_data
def process_raw_data(uploaded_file):
    df_data = pd.read_csv(uploaded_file, sep=';')
    df_data.columns = df_data.columns.str.strip().str.lower()
    
    if 'sitename-sector' in df_data.columns:
        df_data = df_data.rename(columns={'sitename-sector': 'sitesector'})
        
    df_data['sitesector'] = df_data['sitename'].astype(str) + '-' + df_data['sector'].astype(str)
    df_data['datetime_dt'] = pd.to_datetime(df_data['datetime'], format='%d/%m/%y %H')

    kpi_features = ['avg_active_dl_user_bm_h', 'avg_rrc_user_bm_h', 'dl_volume_bm_h', 
                    'prb_dl_bm_h', 'ul_volume_bm_h', 'user_dl_thp_bm_h']
    
    # Vectorized text parsing to replace commas and percentages safely
    for col in kpi_features + ['latitude', 'longitude', 'azimuth']:
        if col in df_data.columns:
            df_data[col] = df_data[col].astype(str).str.replace(',', '.').str.replace('%', '')
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce')

    # Optimized Scaling: Replaced slow groupby.apply() loop with a vectorized transform
    scaler = RobustScaler()
    norm_kpi_features = []
    target_kpis = ['avg_active_dl_user_bm_h', 'avg_rrc_user_bm_h', 'dl_volume_bm_h', 'prb_dl_bm_h']
    
    for c in target_kpis:
        if c in df_data.columns:
            norm_col_name = f'robust_norm_{c}'
            df_data[norm_col_name] = df_data.groupby('sitesector')[c].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten() if len(x) > 1 else x
            )
            norm_kpi_features.append(norm_col_name)

    # Sector Aggregation Pipeline
    agg_map = {col: 'sum' for col in (kpi_features + norm_kpi_features) if col in df_data.columns}
    agg_map.update({c: 'max' for c in ['latitude', 'longitude', 'azimuth'] if c in df_data.columns})

    band_info = df_data.groupby(['sitename', 'sector'])['band'].unique().reset_index()
    band_info.columns = ['sitename', 'sector', 'all_bands']

    df = df_data.groupby(['datetime', 'sitename', 'sector']).agg(agg_map).reset_index()
    df = df.merge(band_info, on=['sitename', 'sector'], how='left')
    df['num_of_bands'] = df['all_bands'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 1)
    df['sitesector'] = df['sitename'].astype(str) + '-' + df['sector'].astype(str)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%y %H')

    # Feature Engineering
    df = df[df['datetime'] >= '2025-12-29'].copy()
    df['hour_only'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    time_min, time_max = df['datetime'].astype('int64').min(), df['datetime'].astype('int64').max()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_only'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_only'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    df['time_progression'] = (df['datetime'].astype('int64') - time_min) / (time_max - time_min)
    df['is_night_hour'] = ((df['hour_only'] >= 0) & (df['hour_only'] <= 6)).astype(int)

    return df, norm_kpi_features, time_min, time_max


#CACHED STEP 2: SAVE, LOAD, AND TRACK ML MODEL AS A RESOURCE
@st.cache_resource
def get_or_train_model(df, train_features, model_path=MODEL_PATH):
    cutoff_data_end = df['datetime'].max()
    cutoff_train_end = cutoff_data_end - pd.Timedelta(days=14)
    
    # Check if pre-existing model is saved on local storage disk
    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
        return model, cutoff_train_end, cutoff_data_end

    # Fallback: Train model if file path missing
    X_train = df[df['datetime'] <= cutoff_train_end][train_features]
    y_train = df[df['datetime'] <= cutoff_train_end]['robust_norm_dl_volume_bm_h']

    train_data = lgb.Dataset(X_train, label=y_train)
    params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1}
    
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # Permanently save back to disk file path
    model.save_model(model_path)
    
    return model, cutoff_train_end, cutoff_data_end


#APP MAIN EXECUTION
if u_raw_data:
    # 1. Fire cached operations
    df, norm_kpi_features, time_min, time_max = process_raw_data(u_raw_data)
    
    geo_features = ['latitude', 'longitude', 'azimuth']
    train_features = norm_kpi_features + geo_features + [
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
        'time_progression', 'is_weekend', 'is_night_hour'
    ]
    
    # 2. Get model (Handles loading disk asset or training + saving dynamically)
    model, cutoff_train_end, cutoff_data_end = get_or_train_model(df, train_features, MODEL_PATH)
    
    # 3. Calculate baseline metrics
    sector_hourly_profile = df[df['datetime'] <= cutoff_train_end].groupby(
        ['sitesector', 'hour_only', 'is_weekend']
    )[norm_kpi_features].mean().reset_index()

    sector = st.selectbox("Select Site-Sector to Visualize:", sorted(df['sitesector'].unique()))

    #FRAGMENT LAYER FOR INTERACTIVE CHART RENDERING
    @st.fragment
    def render_analysis_chart(sector, sla_p, energy_p, time_min, time_max):
        ratio = energy_p / max(sla_p, 0.01)
        sector_data = df[df['sitesector'] == sector].copy().sort_values('datetime')
        this_profile = sector_hourly_profile[sector_hourly_profile['sitesector'] == sector]
        
        sector_mask = df['sitesector'] == sector
        if sector_mask.any():
            raw_val = df.loc[sector_mask, 'all_bands'].iloc[0]
            if isinstance(raw_val, (list, np.ndarray, pd.Series)):
                all_bands = [str(b) for b in raw_val if pd.notna(b)]
            elif pd.notna(raw_val):
                raw_str = str(raw_val).strip()
                delim = '&' if '&' in raw_str else ','
                all_bands = [b.strip() for b in raw_str.split(delim) if b.strip()]
            else:
                all_bands = []
        else:
            all_bands = []

        all_bands = sorted(list(set(all_bands)), key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0), reverse=True)

        def get_dqn_action_detailed(p, bands, ratio):
            t_wake, t_l1, t_l2 = 0.6 * ratio, 0.35 * ratio, 0.15 * ratio
            n = len(bands)
            if n == 0: return "rgba(0,0,0,0)", "No Data", [], []
            if p > t_wake: return "rgba(255, 182, 193, 0.0)", f"Wake Up (0/{n} Sleeping)", bands, []
            elif p > t_l1: return "rgba(182, 149, 192, 0.3)", f"Level 1 Sleep (1/{n} Sleeping)", bands[1:], bands[:1]
            elif p > t_l2:
                mid = max(1, n // 2)
                return "rgba(182, 149, 192, 0.6)", f"Level 2 Sleep ({mid}/{n} Sleeping)", bands[mid:], bands[:mid]
            else: return "rgba(182, 149, 192, 0.9)", f"Max Saving ({n-1}/{n} Sleeping)", bands[-1:], bands[:-1]

        # Forecast Calculations
        backtest_actuals = sector_data[(sector_data['datetime'] > cutoff_train_end) & (sector_data['datetime'] <= cutoff_data_end)].copy()
        backtest_preds = model.predict(backtest_actuals[train_features])
        
        future_dates = pd.date_range(start=cutoff_data_end + pd.Timedelta(hours=1), periods=336, freq='h')
        future_df = pd.DataFrame({'datetime': future_dates, 'hour_only': future_dates.hour, 'day_of_week': future_dates.dayofweek})
        future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)
        future_df['is_night_hour'] = ((future_df['hour_only'] >= 0) & (future_df['hour_only'] <= 6)).astype(int)
        
        for col in geo_features: 
            future_df[col] = sector_data[col].iloc[0]
            
        future_df = future_df.merge(this_profile.drop(columns='sitesector'), on=['hour_only', 'is_weekend'], how='left')
        future_df[norm_kpi_features] = future_df[norm_kpi_features].ffill().bfill().fillna(df[norm_kpi_features].mean())
        future_df['hour_sin'] = np.sin(2 * np.pi * future_df['hour_only'] / 24.0)
        future_df['hour_cos'] = np.cos(2 * np.pi * future_df['hour_only'] / 24.0)
        future_df['day_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7.0)
        future_df['day_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7.0)
        future_df['time_progression'] = (future_df['datetime'].astype('int64') - time_min) / (time_max - time_min)
        lgbm_forecast = model.predict(future_df[train_features])
        
        all_y_values = np.concatenate([sector_data['robust_norm_dl_volume_bm_h'].dropna().values, backtest_preds, lgbm_forecast])
        y_margin = (all_y_values.max() - all_y_values.min()) * 0.05
        y_lo, y_hi = all_y_values.min() - y_margin, all_y_values.max() + y_margin
 
        shapes, back_h = [], []
        for i, p in enumerate(backtest_preds):
            color, action, act, slp = get_dqn_action_detailed(p, all_bands, ratio)
            active_s = ", ".join(act) if act else "None"
            sleep_s = ", ".join(slp) if slp else "None"
            back_h.append(f"Time: {backtest_actuals['datetime'].iloc[i]}<br>Pred: {p:.2f}<br>Action: {action}<br><b>Active:</b> {active_s}<br><b>Sleep:</b> {sleep_s}")
            if i < len(backtest_preds) - 1:
                shapes.append(dict(type="rect", x0=backtest_actuals['datetime'].iloc[i], x1=backtest_actuals['datetime'].iloc[i+1], y0=y_lo, y1=y_hi, fillcolor=color, layer="below", line_width=0))
 
        fore_h = []
        for i, p in enumerate(lgbm_forecast):
            color, action, act, slp = get_dqn_action_detailed(p, all_bands, ratio)
            active_s = ", ".join(act) if act else "None"
            sleep_s = ", ".join(slp) if slp else "None"
            fore_h.append(f"Time: {future_dates[i]}<br>Pred: {p:.2f}<br>Action: {action}<br><b>Active:</b> {active_s}<br><b>Sleep:</b> {sleep_s}")
            if i < len(lgbm_forecast) - 1:
                shapes.append(dict(type="rect", x0=future_dates[i], x1=future_dates[i+1], y0=y_lo, y1=y_hi, fillcolor=color, layer="below", line_width=0))

        # Build Interactive Figure
        fig = go.Figure()
        legend_labels = [
            ("rgba(182, 149, 192, 0.3)", 'DQN: L1 Sleep'),
            ("rgba(182, 149, 192, 0.6)", 'DQN: L2 Sleep'),
            ("rgba(182, 149, 192, 0.9)", 'DQN: Max Saving')
        ]
        for color, label in legend_labels:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=12, symbol='square', color=color), name=label))

        fig.add_trace(go.Scatter(x=sector_data['datetime'], y=sector_data['dl_volume_bm_h'], name='Real Traffic (GB)', line=dict(color='black', width=2), yaxis='y2'))
        fig.add_trace(go.Scatter(x=sector_data['datetime'], y=sector_data['robust_norm_dl_volume_bm_h'], name='Norm. Historical Data', line=dict(color='rgba(52, 152, 219, 0.3)')))
        fig.add_trace(go.Scatter(x=backtest_actuals['datetime'], y=backtest_preds, name='Backtest', line=dict(color='#e74c3c', width=2), text=back_h, hovertemplate="%{text}<extra></extra>"))
        fig.add_trace(go.Scatter(x=future_dates, y=lgbm_forecast, name='LGBM Forecast', line=dict(color='#27ae60', width=3), text=fore_h, hovertemplate="%{text}<extra></extra>"))

        fig.update_layout(shapes=shapes, template='plotly_white', height=750, uirevision=True,
                            xaxis=dict(rangeslider=dict(visible=True), type="date"),
                            yaxis=dict(title="Normalized Scale (LGBM)"),
                            yaxis2=dict(title="Real Traffic (GB)", side="right", overlaying="y", anchor="x", showgrid=False),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        st.plotly_chart(fig, use_container_width=True, key="network_main_chart")

        # Report Exporter
        html_bytes = fig.to_html(include_plotlyjs='cdn').encode()
        st.download_button(label="Download Analysis Report (HTML)", data=html_bytes, file_name=f"Network_Report_{sector}.html", mime="text/html")

        # Metric Displays
        y_true = backtest_actuals['robust_norm_dl_volume_bm_h'].values
        p_f1 = f1_score((y_true > 0.4).astype(int), (backtest_preds > 0.4).astype(int), zero_division=0)
        s_f1 = f1_score((y_true <= 0.1).astype(int), (backtest_preds <= 0.1).astype(int), zero_division=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Peak F1", f"{p_f1*100:.1f}%")
        c2.metric("Sleep F1", f"{s_f1*100:.1f}%")
        c3.write(f"**Status:** Ratio={ratio:.2f} | Sector: {sector} ({len(all_bands)} Cells)")

    render_analysis_chart(sector, sla_p, energy_p, time_min, time_max)
else:
    st.info("Please upload the required files in the sidebar.")