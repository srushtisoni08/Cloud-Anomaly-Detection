import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import streamlit as st
from datetime import datetime, timedelta

class SecurityDashboard:
    """
    Advanced visualization components for the security dashboard.
    """
    
    def __init__(self):
        self.color_palette = {
            'high_risk': '#FF4B4B',
            'medium_risk': '#FF8C00', 
            'low_risk': '#32CD32',
            'normal': '#1f77b4',
            'background': '#f8f9fa'
        }
    
    def create_risk_distribution_chart(self, data, risk_scores):
        """Create risk score distribution visualization."""
        fig = go.Figure()
        
        # Risk score histogram
        fig.add_trace(go.Histogram(
            x=risk_scores,
            nbinsx=30,
            name='Risk Distribution',
            marker_color='rgba(55, 128, 191, 0.7)',
            hovertemplate='Risk Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add threshold lines
        fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Risk Threshold")
        
        fig.update_layout(
            title="Login Risk Score Distribution",
            xaxis_title="Risk Score",
            yaxis_title="Number of Logins",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_temporal_heatmap(self, data):
        """Create temporal pattern heatmap showing login activity by hour and day."""
        # Create hour vs day of week matrix
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['day_name'] = pd.to_datetime(data['timestamp']).dt.day_name()
        
        # Create pivot table
        heatmap_data = data.groupby(['day_name', 'hour']).size().reset_index(name='count')
        heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='count').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(day_order)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=list(range(24)),
            y=day_order,
            colorscale='YlOrRd',
            hoverongaps=False,
            hovertemplate='Hour: %{x}<br>Day: %{y}<br>Login Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Login Activity Heatmap (Day vs Hour)",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            height=400
        )
        
        return fig
    
    def create_geographic_risk_map(self, data, risk_scores):
        """Create an interactive map showing geographic risk distribution."""
        # Add risk scores to data
        map_data = data.copy()
        map_data['risk_score'] = risk_scores
        map_data['risk_level'] = pd.cut(
            risk_scores,
            bins=[0, 0.5, 0.8, 1],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Calculate center point
        center_lat = map_data['latitude'].mean()
        center_lon = map_data['longitude'].mean()
        
        # Create base map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
        
        # Add marker clusters for different risk levels
        high_risk_cluster = MarkerCluster(name="High Risk Logins").add_to(m)
        medium_risk_cluster = MarkerCluster(name="Medium Risk Logins").add_to(m)
        low_risk_cluster = MarkerCluster(name="Low Risk Logins").add_to(m)
        
        # Color mapping
        risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        
        # Add markers to appropriate clusters
        for idx, row in map_data.iterrows():
            popup_text = f"""
            User: {row['user_id']}<br>
            Time: {row['timestamp']}<br>
            Location: {row['city']}, {row['country']}<br>
            Risk Score: {row['risk_score']:.3f}<br>
            Risk Level: {row['risk_level']}<br>
            Device: {row['device_type']}<br>
            IP: {row['ip_address']}
            """
            
            marker = folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color=risk_colors[row['risk_level']],
                fill=True,
                fillColor=risk_colors[row['risk_level']],
                fillOpacity=0.8
            )
            
            if row['risk_level'] == 'High':
                marker.add_to(high_risk_cluster)
            elif row['risk_level'] == 'Medium':
                marker.add_to(medium_risk_cluster)
            else:
                marker.add_to(low_risk_cluster)
        
        # Add heatmap overlay for login density
        heat_data = [[row['latitude'], row['longitude'], row['risk_score']] 
                     for idx, row in map_data.iterrows()]
        
        HeatMap(heat_data, name="Risk Density Heatmap", 
                min_opacity=0.2, radius=15, blur=10).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_velocity_analysis_chart(self, data):
        """Create visualization for impossible travel detection."""
        # Filter out zero velocities and extreme outliers for better visualization
        velocity_data = data[data['velocity_kmh'] > 0].copy()
        velocity_data = velocity_data[velocity_data['velocity_kmh'] < 5000]  # Remove extreme outliers
        
        if len(velocity_data) == 0:
            # Return empty figure if no valid velocity data
            fig = go.Figure()
            fig.add_annotation(text="No velocity data available", 
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Login Velocity Analysis", height=400)
            return fig
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Velocity Distribution', 'Velocity Timeline'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # Velocity histogram
        fig.add_trace(
            go.Histogram(
                x=velocity_data['velocity_kmh'],
                nbinsx=50,
                name='Velocity Distribution',
                marker_color='rgba(55, 128, 191, 0.7)',
                hovertemplate='Velocity: %{x:.1f} km/h<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add reference lines for travel speeds
        speeds = [
            (100, 'Car Speed', 'green'),
            (300, 'High Speed Train', 'blue'),
            (900, 'Commercial Aircraft', 'orange'),
            (1200, 'Suspicious Speed', 'red')
        ]
        
        for speed, label, color in speeds:
            fig.add_vline(x=speed, line_dash="dash", line_color=color, 
                         annotation_text=label, row=1, col=1)
        
        # Velocity over time (sample recent data to avoid clutter)
        recent_data = velocity_data.tail(100).sort_values('timestamp')
        
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(recent_data['timestamp']),
                y=recent_data['velocity_kmh'],
                mode='markers',
                name='Recent Velocities',
                marker=dict(
                    size=6,
                    color=recent_data['velocity_kmh'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Velocity (km/h)")
                ),
                hovertemplate='Time: %{x}<br>Velocity: %{y:.1f} km/h<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Login Velocity Analysis - Impossible Travel Detection",
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Velocity (km/h)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Velocity (km/h)", row=2, col=1)
        
        return fig
    
    def create_device_analysis_chart(self, data):
        """Create comprehensive device analysis visualization."""
        # Create subplot for multiple device metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Device Types', 'Browser Distribution', 
                          'Operating Systems', 'Device Combinations'),
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Device types
        device_counts = data['device_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=device_counts.index, values=device_counts.values,
                   name="Device Types", hole=0.3),
            row=1, col=1
        )
        
        # Browser distribution
        browser_counts = data['browser'].value_counts().head(8)
        fig.add_trace(
            go.Pie(labels=browser_counts.index, values=browser_counts.values,
                   name="Browsers", hole=0.3),
            row=1, col=2
        )
        
        # Operating systems
        os_counts = data['operating_system'].value_counts().head(8)
        fig.add_trace(
            go.Pie(labels=os_counts.index, values=os_counts.values,
                   name="Operating Systems", hole=0.3),
            row=2, col=1
        )
        
        # Device combinations per user (unusual device detection)
        user_device_counts = data.groupby('user_id').agg({
            'device_type': 'nunique',
            'browser': 'nunique', 
            'operating_system': 'nunique'
        }).reset_index()
        
        user_device_counts['total_combinations'] = (
            user_device_counts['device_type'] + 
            user_device_counts['browser'] + 
            user_device_counts['operating_system']
        )
        
        # Users with many different devices might be suspicious
        combo_dist = user_device_counts['total_combinations'].value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(x=combo_dist.index, y=combo_dist.values,
                   name="Device Combinations",
                   marker_color='rgba(55, 128, 191, 0.7)',
                   hovertemplate='Combinations: %{x}<br>Users: %{y}<extra></extra>'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Device and Browser Analysis",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_user_behavior_timeline(self, data, selected_user=None):
        """Create detailed timeline for specific user behavior analysis."""
        if selected_user:
            user_data = data[data['user_id'] == selected_user].copy()
        else:
            # Show aggregate timeline
            user_data = data.copy()
        
        if len(user_data) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data for selected user", 
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="User Behavior Timeline", height=400)
            return fig
        
        user_data = user_data.sort_values('timestamp')
        user_data['timestamp_dt'] = pd.to_datetime(user_data['timestamp'])
        
        # Create timeline with multiple traces
        fig = go.Figure()
        
        # Login success/failure
        success_data = user_data[user_data['success'] == True]
        failed_data = user_data[user_data['success'] == False]
        
        if len(success_data) > 0:
            fig.add_trace(go.Scatter(
                x=success_data['timestamp_dt'],
                y=[1] * len(success_data),
                mode='markers',
                name='Successful Logins',
                marker=dict(color='green', size=8, symbol='circle'),
                hovertemplate='%{x}<br>Status: Success<br>Location: %{text}<extra></extra>',
                text=[f"{row['city']}, {row['country']}" for _, row in success_data.iterrows()]
            ))
        
        if len(failed_data) > 0:
            fig.add_trace(go.Scatter(
                x=failed_data['timestamp_dt'],
                y=[0.5] * len(failed_data),
                mode='markers',
                name='Failed Logins',
                marker=dict(color='red', size=8, symbol='x'),
                hovertemplate='%{x}<br>Status: Failed<br>Location: %{text}<extra></extra>',
                text=[f"{row['city']}, {row['country']}" for _, row in failed_data.iterrows()]
            ))
        
        # Add device change indicators
        device_changes = user_data[user_data['device_type'] != user_data['device_type'].shift(1)]
        
        if len(device_changes) > 1:  # Skip first row which will always be a "change"
            device_changes = device_changes.iloc[1:]
            fig.add_trace(go.Scatter(
                x=device_changes['timestamp_dt'],
                y=[1.5] * len(device_changes),
                mode='markers',
                name='Device Changes',
                marker=dict(color='orange', size=10, symbol='diamond'),
                hovertemplate='%{x}<br>Device Change<br>New Device: %{text}<extra></extra>',
                text=[f"{row['device_type']} - {row['browser']}" for _, row in device_changes.iterrows()]
            ))
        
        fig.update_layout(
            title=f"Timeline for User: {selected_user}" if selected_user else "Overall Login Timeline",
            xaxis_title="Time",
            yaxis_title="Event Type",
            yaxis=dict(
                tickmode='array',
                tickvals=[0.5, 1, 1.5],
                ticktext=['Failed', 'Success', 'Device Change']
            ),
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_risk_trend_chart(self, data, risk_scores, window_hours=24):
        """Create risk trend analysis over time."""
        # Add risk scores to data
        trend_data = data.copy()
        trend_data['risk_score'] = risk_scores
        trend_data['timestamp_dt'] = pd.to_datetime(trend_data['timestamp'])
        
        # Group by time windows and calculate average risk
        trend_data['time_window'] = trend_data['timestamp_dt'].dt.floor(f'{window_hours}H')
        
        risk_trends = trend_data.groupby('time_window').agg({
            'risk_score': ['mean', 'max', 'count'],
            'user_id': 'nunique'
        }).reset_index()
        
        risk_trends.columns = ['time_window', 'avg_risk', 'max_risk', 'login_count', 'unique_users']
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Risk Score Trends', 'Login Activity'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Risk trends
        fig.add_trace(
            go.Scatter(
                x=risk_trends['time_window'],
                y=risk_trends['avg_risk'],
                mode='lines+markers',
                name='Average Risk',
                line=dict(color='blue', width=2),
                hovertemplate='Time: %{x}<br>Avg Risk: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=risk_trends['time_window'],
                y=risk_trends['max_risk'],
                mode='lines+markers',
                name='Maximum Risk',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Time: %{x}<br>Max Risk: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=0.8, line_dash="dot", line_color="red", 
                     annotation_text="High Risk", row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="orange", 
                     annotation_text="Medium Risk", row=1, col=1)
        
        # Login activity
        fig.add_trace(
            go.Bar(
                x=risk_trends['time_window'],
                y=risk_trends['login_count'],
                name='Login Count',
                marker_color='rgba(55, 128, 191, 0.7)',
                hovertemplate='Time: %{x}<br>Logins: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Security Risk Trends Over Time",
            height=600,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Risk Score", row=1, col=1)
        fig.update_yaxes(title_text="Login Count", row=2, col=1)
        
        return fig
    
    def create_alert_dashboard(self, alerts):
        """Create comprehensive alert monitoring dashboard."""
        if not alerts:
            fig = go.Figure()
            fig.add_annotation(text="No alerts to display", 
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Security Alerts Dashboard", height=400)
            return fig
        
        alerts_df = pd.DataFrame(alerts)
        alerts_df['timestamp_dt'] = pd.to_datetime(alerts_df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Alert Levels Over Time', 'Risk Factor Frequency',
                          'Top Affected Users', 'Alert Volume by Hour'),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Alert levels over time
        for level in ['HIGH', 'MEDIUM']:
            level_data = alerts_df[alerts_df['alert_level'] == level]
            if len(level_data) > 0:
                # Group by hour for better visualization
                hourly_counts = level_data.set_index('timestamp_dt').resample('H').size()
                fig.add_trace(
                    go.Scatter(
                        x=hourly_counts.index,
                        y=hourly_counts.values,
                        mode='lines+markers',
                        name=f'{level} Risk Alerts',
                        line=dict(color='red' if level == 'HIGH' else 'orange')
                    ),
                    row=1, col=1
                )
        
        # Risk factor frequency
        risk_factors = {}
        for alert in alerts:
            factors = alert.get('risk_factors', {})
            for factor, score in factors.items():
                if score > 0.3 and factor != 'total_additional_risk':
                    risk_factors[factor] = risk_factors.get(factor, 0) + 1
        
        if risk_factors:
            fig.add_trace(
                go.Bar(
                    x=list(risk_factors.keys()),
                    y=list(risk_factors.values()),
                    name='Risk Factors',
                    marker_color='rgba(255, 99, 71, 0.7)'
                ),
                row=1, col=2
            )
        
        # Top affected users
        user_counts = alerts_df['user_id'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=user_counts.index,
                y=user_counts.values,
                name='Affected Users',
                marker_color='rgba(255, 165, 0, 0.7)'
            ),
            row=2, col=1
        )
        
        # Alert volume by hour
        alerts_df['hour'] = alerts_df['timestamp_dt'].dt.hour
        hourly_alert_counts = alerts_df['hour'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=hourly_alert_counts.index,
                y=hourly_alert_counts.values,
                name='Hourly Alerts',
                marker_color='rgba(60, 179, 113, 0.7)'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Security Alerts Dashboard",
            height=600,
            showlegend=False
        )
        
        return fig
