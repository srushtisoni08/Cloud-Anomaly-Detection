import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from flask import Flask, render_template
from data_simulator import LoginDataSimulator
from ml_models import AnomalyDetectionModels
from anomaly_detector import RealTimeAnomalyDetector
from visualizations import SecurityDashboard
from utils import load_sample_data, calculate_risk_metrics

st.markdown(
    """
    <style>
    /* App background */
    .stApp, body {
        background: linear-gradient(135deg, #071021 0%, #0A1633 40%, #001B3A 70%, #002A4A 100%) !important;
        color: #FFFFFF;
    }

    /* Header (top bar) */
    header[data-testid="stHeader"] {
        background: transparent !important;
        color: #FFFFFF !important;
    }
    header[data-testid="stHeader"] * { color: #FFFFFF !important; }

    /* Sidebar (navigation) */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #071021 0%, #0A1633 40%, #001B3A 70%, #002A4A 100%) !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }

    /* Plotly containers: remove white cards */
    div[data-testid="stPlotlyChart"] { background: transparent !important; }

    /* Detailed Logs: style boxes (filters and table) */
    div[data-testid="stSelectbox"] {
        background: rgba(10, 22, 51, 0.35) !important;
        border: 1px solid rgba(75, 155, 224, 0.35) !important;
        border-radius: 8px !important;
        padding: 6px !important;
    }
    div[data-testid="stSelectbox"] input { background: transparent !important; color: #000000 !important; }
    div[data-baseweb="select"] div { color: #000000 !important; }

    div[data-testid="stDataFrame"] {
        background: rgba(10, 22, 51, 0.35) !important;
        
        border: 1px solid rgba(75, 155, 224, 0.35) !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
    div[data-testid="stDataFrame"] * { color: #E6F7FF !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if 'login_data' not in st.session_state:
    st.session_state.login_data = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'detector' not in st.session_state:
    st.session_state.detector = None

def main():
    st.title("🛡️ AI-Powered Login Anomaly Detection System")
    st.markdown("### Protecting cloud environments from credential theft and account takeovers")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Data simulation parameters
        st.subheader("Data Generation")
        num_users = st.slider("Number of Users", 10, 1000, 100)
        num_days = st.slider("Data History (Days)", 7, 90, 30)
        anomaly_rate = st.slider("Anomaly Rate (%)", 1, 20, 5)
        
        # Model configuration
        st.subheader("ML Model Settings")
        contamination = st.slider("Contamination Factor", 0.01, 0.3, 0.1, 0.01)
        n_estimators = st.slider("Number of Estimators", 50, 200, 100, 10)
        
        # Detection thresholds
        st.subheader("Alert Thresholds")
        high_risk_threshold = st.slider("High Risk Threshold", 0.5, 0.95, 0.8, 0.05)
        medium_risk_threshold = st.slider("Medium Risk Threshold", 0.3, 0.7, 0.5, 0.05)
        
        # Generate data button
        if st.button("🔄 Generate New Data", type="primary"):
            with st.spinner("Generating login data..."):
                simulator = LoginDataSimulator()
                st.session_state.login_data = simulator.generate_login_data(
                    num_users=num_users,
                    num_days=num_days,
                    anomaly_rate=anomaly_rate/100
                )
                st.success("Data generated successfully!")
        
        # Train models button
        if st.button("🤖 Train ML Models", type="primary") and st.session_state.login_data is not None:
            with st.spinner("Training anomaly detection models..."):
                models = AnomalyDetectionModels()
                models.train_models(
                    st.session_state.login_data,
                    contamination=contamination,
                    n_estimators=n_estimators
                )
                st.session_state.models = models
                
                # Initialize real-time detector
                st.session_state.detector = RealTimeAnomalyDetector(
                    models,
                    high_risk_threshold=high_risk_threshold,
                    medium_risk_threshold=medium_risk_threshold
                )
                st.success("Models trained successfully!")
    
    # Main content area
    if st.session_state.login_data is None:
        st.markdown(
            """
            <div style="background: rgba(10, 22, 51, 0.35); border-left: 4px solid #4B9BE0; padding: 0.75rem 1rem; border-radius: 6px; color:#F5FAFF;">
            👆 Please configure parameters and generate data using the sidebar controls.
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Show sample interface preview
        st.subheader("📊 System Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Logins", "0", "Generate data to see metrics")
        with col2:
            st.metric("Anomalies Detected", "0", "0%")
        with col3:
            st.metric("High Risk Events", "0")
        with col4:
            st.metric("Users Monitored", "0")
        
        return
    
    # Display system metrics
    data = st.session_state.login_data
    total_logins = len(data)
    unique_users = data['user_id'].nunique()
    
    if st.session_state.models is not None:
        predictions = st.session_state.models.predict_anomalies(data)
        anomalies = sum(predictions == -1)
        high_risk_events = len(data[data['risk_score'] > high_risk_threshold]) if 'risk_score' in data.columns else 0
    else:
        anomalies = 0
        high_risk_events = 0
    
    # Metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Logins", f"{total_logins:,}", f"{total_logins - 0:+,}")
    with col2:
        anomaly_rate_display = (anomalies / total_logins * 100) if total_logins > 0 else 0
        st.metric("Anomalies Detected", f"{anomalies:,}", f"{anomaly_rate_display:.1f}%")
    with col3:
        st.metric("High Risk Events", f"{high_risk_events:,}")
    with col4:
        st.metric("Users Monitored", f"{unique_users:,}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Geographic Analysis", 
        "⏰ Temporal Patterns", 
        "📱 Device Analysis", 
        "🚨 Real-time Detection",
        "📋 Detailed Logs"
    ])
    
    with tab1:
        st.subheader("Geographic Login Distribution")
        
        if st.session_state.models is not None:
            # Add risk scores to data
            risk_scores = st.session_state.models.calculate_risk_scores(data)
            data_with_scores = data.copy()
            data_with_scores['risk_score'] = risk_scores
            data_with_scores['risk_level'] = pd.cut(
                risk_scores,
                bins=[0, medium_risk_threshold, high_risk_threshold, 1],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        else:
            data_with_scores = data.copy()
            data_with_scores['risk_level'] = 'Unknown'
        
        # Create Folium map
        center_lat = data_with_scores['latitude'].mean()
        center_lon = data_with_scores['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
        
        # Color mapping for risk levels
        color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Unknown': 'blue'}
        
        # Add markers for recent logins (last 100 to avoid clutter)
        recent_logins = data_with_scores.tail(100)
        
        for idx, row in recent_logins.iterrows():
            color = color_map.get(row.get('risk_level', 'Unknown'), 'blue')
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=f"User: {row['user_id']}<br>Time: {row['timestamp']}<br>City: {row['city']}<br>Risk: {row.get('risk_level', 'Unknown')}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)
        
        # Geographic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Countries")
            country_stats = data['country'].value_counts().head(10)
            fig = px.bar(
                x=country_stats.values,
                y=country_stats.index,
                orientation='h',
                labels={'x': 'Login Count', 'y': 'Country'},
                title="Login Distribution by Country"
            )
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top Cities")
            city_stats = data['city'].value_counts().head(10)
            fig2 = px.bar(
                x=city_stats.values,
                y=city_stats.index,
                orientation='h',
                labels={'x': 'Login Count', 'y': 'City'},
                title="Login Distribution by City"
            )
            fig2.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Temporal Login Patterns")
        # Time series of logins per day
        daily_logins = data.resample('D', on='timestamp').size()
        fig_ts = px.line(daily_logins, labels={'value': 'Logins', 'timestamp': 'Date'}, title="Logins Over Time")
        fig_ts.update_layout(
            font=dict(color="#ffffff"),
            title_font=dict(color="#ffffff"),
            xaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
            yaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Hour-of-day distribution
        data['hour'] = data['timestamp'].dt.hour
        hourly_counts = data['hour'].value_counts().sort_index()
        fig_hour = px.bar(x=hourly_counts.index, y=hourly_counts.values, labels={'x': 'Hour', 'y': 'Logins'}, title="Logins by Hour")
        fig_hour.update_layout(
            font=dict(color="#ffffff"),
            title_font=dict(color="#ffffff"),
            xaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
            yaxis=dict(title_font=dict(color="#ffffff"), tickfont=dict(color="#ffffff")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    
    with tab3:
        st.subheader("Device and Browser Analysis")
        device_counts = data['device_type'].value_counts()
        browser_counts = data['browser'].value_counts()
        os_counts = data['operating_system'].value_counts()
        
        colA, colB, colC = st.columns(3)
        with colA:
            fig_dev = px.pie(values=device_counts.values, names=device_counts.index, title="Devices")
            fig_dev.update_layout(title_font=dict(color="#FFFFFF"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dev, use_container_width=True)
        with colB:
            fig_bro = px.pie(values=browser_counts.values, names=browser_counts.index, title="Browsers")
            fig_bro.update_layout(title_font=dict(color="#FFFFFF"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bro, use_container_width=True)
        with colC:
            fig_os = px.pie(values=os_counts.values, names=os_counts.index, title="Operating Systems")
            fig_os.update_layout(title_font=dict(color="#FFFFFF"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_os, use_container_width=True)
    
    with tab4:
        st.subheader("Real-time Anomaly Detection")
        if st.session_state.detector is not None:
            # Simulate a real-time risk feed visualization
            latest_window = data.tail(200)
            risk_scores = st.session_state.models.calculate_risk_scores(latest_window)
            fig_rt = px.histogram(risk_scores, nbins=30, title="Risk Score Distribution (Latest Window)")
            fig_rt.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_rt, use_container_width=True)
        else:
            st.info("Train the models to enable real-time detection.")
    
    with tab5:
        st.subheader("Detailed Login Logs")
        # Filters
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            country_filter = st.selectbox("Country", options=["All"] + sorted(data['country'].unique().tolist()))
        with colf2:
            device_filter = st.selectbox("Device", options=["All"] + sorted(data['device_type'].unique().tolist()))
        with colf3:
            browser_filter = st.selectbox("Browser", options=["All"] + sorted(data['browser'].unique().tolist()))
        
        # Apply filters
        filtered_data = data.copy()
        if country_filter != "All":
            filtered_data = filtered_data[filtered_data['country'] == country_filter]
        if device_filter != "All":
            filtered_data = filtered_data[filtered_data['device_type'] == device_filter]
        if browser_filter != "All":
            filtered_data = filtered_data[filtered_data['browser'] == browser_filter]
        
        # Display filtered data
        st.write(f"Showing {len(filtered_data)} of {len(data)} login records")
        
        # Select columns to display
        display_columns = ['timestamp', 'user_id', 'ip_address', 'city', 'country', 
                         'device_type', 'browser', 'operating_system']
        
        if st.session_state.models is not None and 'risk_score' in filtered_data.columns:
            display_columns.append('risk_score')
        
        st.dataframe(
            filtered_data[display_columns].sort_values('timestamp', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Export functionality
        if st.button("📥 Export Filtered Data"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"login_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)