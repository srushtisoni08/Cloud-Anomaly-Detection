import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

from data_simulator import LoginDataSimulator
from ml_models import AnomalyDetectionModels
from anomaly_detector import RealTimeAnomalyDetector
from visualizations import SecurityDashboard
from utils import load_sample_data, calculate_risk_metrics

st.set_page_config(
    page_title="AI Login Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
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
        if st.button("🤖 Train ML Models") and st.session_state.login_data is not None:
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
        st.info("👆 Please configure parameters and generate data using the sidebar controls.")
        
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
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top Cities")
            city_stats = data['city'].value_counts().head(10)
            fig = px.bar(
                x=city_stats.values,
                y=city_stats.index,
                orientation='h',
                labels={'x': 'Login Count', 'y': 'City'},
                title="Login Distribution by City"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Temporal Login Patterns")
        
        # Convert timestamp to datetime if it's not already
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.day_name()
        data['date'] = data['timestamp'].dt.date
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly distribution
            hourly_logins = data['hour'].value_counts().sort_index()
            fig = px.bar(
                x=hourly_logins.index,
                y=hourly_logins.values,
                labels={'x': 'Hour of Day', 'y': 'Login Count'},
                title="Login Distribution by Hour"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week distribution
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_logins = data['day_of_week'].value_counts().reindex(day_order)
            fig = px.bar(
                x=daily_logins.index,
                y=daily_logins.values,
                labels={'x': 'Day of Week', 'y': 'Login Count'},
                title="Login Distribution by Day"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series plot
        st.subheader("Login Timeline")
        daily_counts = data.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Daily Login Volume",
            labels={'count': 'Number of Logins', 'date': 'Date'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Device and Browser Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Device distribution
            device_stats = data['device_type'].value_counts()
            fig = px.pie(
                values=device_stats.values,
                names=device_stats.index,
                title="Device Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Browser distribution
            browser_stats = data['browser'].value_counts().head(8)
            fig = px.pie(
                values=browser_stats.values,
                names=browser_stats.index,
                title="Browser Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # OS distribution
            os_stats = data['operating_system'].value_counts().head(8)
            fig = px.pie(
                values=os_stats.values,
                names=os_stats.index,
                title="Operating System Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Device fingerprint analysis
        st.subheader("Unusual Device Detection")
        if st.session_state.models is not None:
            device_features = ['device_type', 'browser', 'operating_system']
            unusual_devices = st.session_state.models.detect_unusual_devices(data, device_features)
            
            if len(unusual_devices) > 0:
                st.warning(f"⚠️ Detected {len(unusual_devices)} unusual device combinations")
                st.dataframe(unusual_devices[['user_id', 'timestamp', 'device_type', 'browser', 'operating_system', 'city']])
            else:
                st.success("✅ No unusual device patterns detected")
        else:
            st.info("Train ML models to enable unusual device detection")
    
    with tab4:
        st.subheader("Real-time Anomaly Detection")
        
        if st.session_state.detector is not None:
            # Real-time detection simulation
            st.subheader("🔴 Live Detection Feed")
            
            # Show recent high-risk events
            recent_data = data.tail(50).copy()
            if st.session_state.models is not None:
                risk_scores = st.session_state.models.calculate_risk_scores(recent_data)
                recent_data['risk_score'] = risk_scores
                recent_data['alert_level'] = recent_data['risk_score'].apply(
                    lambda x: 'HIGH' if x > high_risk_threshold else 'MEDIUM' if x > medium_risk_threshold else 'LOW'
                )
                
                # Filter and display alerts
                alerts = recent_data[recent_data['risk_score'] > medium_risk_threshold].copy()
                alerts = alerts.sort_values('risk_score', ascending=False)
                
                if len(alerts) > 0:
                    for idx, alert in alerts.head(10).iterrows():
                        alert_color = "🔴" if alert['alert_level'] == 'HIGH' else "🟡"
                        
                        with st.container():
                            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                            
                            with col1:
                                st.write(f"{alert_color} **{alert['alert_level']}**")
                            
                            with col2:
                                st.write(f"**User:** {alert['user_id']}")
                                st.write(f"**Time:** {alert['timestamp']}")
                            
                            with col3:
                                st.write(f"**Location:** {alert['city']}, {alert['country']}")
                                st.write(f"**Device:** {alert['device_type']}")
                            
                            with col4:
                                st.metric("Risk Score", f"{alert['risk_score']:.3f}")
                            
                            st.markdown("---")
                else:
                    st.success("✅ No recent high-risk login attempts detected")
            else:
                st.info("Train ML models to enable real-time detection")
            
            # Detection statistics
            if st.session_state.models is not None:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_risk_count = len(data[data['risk_score'] > high_risk_threshold]) if 'risk_score' in data.columns else 0
                    st.metric("High Risk Logins", high_risk_count)
                
                with col2:
                    medium_risk_count = len(data[
                        (data['risk_score'] > medium_risk_threshold) & 
                        (data['risk_score'] <= high_risk_threshold)
                    ]) if 'risk_score' in data.columns else 0
                    st.metric("Medium Risk Logins", medium_risk_count)
                
                with col3:
                    low_risk_count = len(data[data['risk_score'] <= medium_risk_threshold]) if 'risk_score' in data.columns else 0
                    st.metric("Low Risk Logins", low_risk_count)
        else:
            st.info("Configure and train models to enable real-time detection")
    
    with tab5:
        st.subheader("Detailed Login Logs")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_users = st.multiselect(
                "Filter by Users",
                options=sorted(data['user_id'].unique()),
                default=[]
            )
        
        with col2:
            selected_countries = st.multiselect(
                "Filter by Countries",
                options=sorted(data['country'].unique()),
                default=[]
            )
        
        with col3:
            if st.session_state.models is not None and 'risk_score' in data.columns:
                risk_filter = st.selectbox(
                    "Risk Level Filter",
                    options=['All', 'High Risk Only', 'Medium Risk Only', 'Low Risk Only']
                )
            else:
                risk_filter = 'All'
        
        # Apply filters
        filtered_data = data.copy()
        
        if selected_users:
            filtered_data = filtered_data[filtered_data['user_id'].isin(selected_users)]
        
        if selected_countries:
            filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]
        
        if st.session_state.models is not None and 'risk_score' in filtered_data.columns:
            if risk_filter == 'High Risk Only':
                filtered_data = filtered_data[filtered_data['risk_score'] > high_risk_threshold]
            elif risk_filter == 'Medium Risk Only':
                filtered_data = filtered_data[
                    (filtered_data['risk_score'] > medium_risk_threshold) & 
                    (filtered_data['risk_score'] <= high_risk_threshold)
                ]
            elif risk_filter == 'Low Risk Only':
                filtered_data = filtered_data[filtered_data['risk_score'] <= medium_risk_threshold]
        
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

if __name__ == "__main__":
    main()
