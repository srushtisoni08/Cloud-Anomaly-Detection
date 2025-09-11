import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Optional, Tuple
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load sample data for demonstration purposes.
    In production, this would connect to actual data sources.
    """
    if file_path and file_path.endswith('.csv'):
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            st.warning(f"File {file_path} not found. Generating sample data instead.")
    
    return pd.DataFrame()

def calculate_risk_metrics(data: pd.DataFrame, risk_scores: np.ndarray) -> Dict:
    """
    Calculate comprehensive risk metrics for dashboard display.
    """
    if len(data) == 0 or len(risk_scores) == 0:
        return {
            'total_logins': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'avg_risk_score': 0.0,
            'risk_distribution': {},
            'temporal_risk_patterns': {},
            'geographic_risk_summary': {},
            'user_risk_summary': {}
        }
    
    total_logins = len(data)
    high_risk_threshold = 0.8
    medium_risk_threshold = 0.5
    
    high_risk_count = int(np.sum(risk_scores > high_risk_threshold))
    medium_risk_count = int(np.sum((risk_scores > medium_risk_threshold) & (risk_scores <= high_risk_threshold)))
    low_risk_count = int(np.sum(risk_scores <= medium_risk_threshold))
    avg_risk_score = float(np.mean(risk_scores))
    
    risk_distribution = {
        'p50': float(np.percentile(risk_scores, 50)),
        'p75': float(np.percentile(risk_scores, 75)),
        'p90': float(np.percentile(risk_scores, 90)),
        'p95': float(np.percentile(risk_scores, 95)),
        'p99': float(np.percentile(risk_scores, 99))
    }
    
    data_with_risk = data.copy()
    data_with_risk['risk_score'] = risk_scores
    data_with_risk['hour'] = pd.to_datetime(data_with_risk['timestamp']).dt.hour
    data_with_risk['day_of_week'] = pd.to_datetime(data_with_risk['timestamp']).dt.day_name()
    
    temporal_risk_patterns = {
        'hourly_risk': data_with_risk.groupby('hour')['risk_score'].mean().to_dict(),
        'daily_risk': data_with_risk.groupby('day_of_week')['risk_score'].mean().to_dict(),
        'peak_risk_hour': int(data_with_risk.groupby('hour')['risk_score'].mean().idxmax()),
        'peak_risk_day': data_with_risk.groupby('day_of_week')['risk_score'].mean().idxmax()
    }

    geographic_risk = data_with_risk.groupby('country').agg({
        'risk_score': ['mean', 'max', 'count']
    }).round(3)
    geographic_risk.columns = ['avg_risk', 'max_risk', 'login_count']
    
    geographic_risk_summary = {
        'high_risk_countries': geographic_risk[geographic_risk['avg_risk'] > high_risk_threshold].to_dict('index'),
        'top_risk_countries': geographic_risk.nlargest(5, 'avg_risk').to_dict('index'),
        'most_active_countries': geographic_risk.nlargest(5, 'login_count').to_dict('index')
    }

    user_risk = data_with_risk.groupby('user_id').agg({
        'risk_score': ['mean', 'max', 'count']
    }).round(3)
    user_risk.columns = ['avg_risk', 'max_risk', 'login_count']
    
    user_risk_summary = {
        'high_risk_users': user_risk[user_risk['avg_risk'] > high_risk_threshold].to_dict('index'),
        'top_risk_users': user_risk.nlargest(10, 'avg_risk').to_dict('index'),
        'most_active_users': user_risk.nlargest(10, 'login_count').to_dict('index')
    }
    
    return {
        'total_logins': total_logins,
        'high_risk_count': high_risk_count,
        'medium_risk_count': medium_risk_count,
        'low_risk_count': low_risk_count,
        'avg_risk_score': avg_risk_score,
        'risk_distribution': risk_distribution,
        'temporal_risk_patterns': temporal_risk_patterns,
        'geographic_risk_summary': geographic_risk_summary,
        'user_risk_summary': user_risk_summary
    }

def format_alert_message(alert: Dict) -> str:
    """
    Format alert information for display in the dashboard.
    """
    risk_level_emoji = {
        'HIGH': '🔴',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    
    emoji = risk_level_emoji.get(alert.get('alert_level', 'LOW'), '🔵')
    
    message = f"""
    {emoji} **{alert.get('alert_level', 'UNKNOWN')} RISK ALERT**
    
    **Alert ID:** {alert.get('alert_id', 'N/A')}
    **Time:** {alert.get('timestamp', 'N/A')}
    **Risk Score:** {alert.get('risk_score', 0):.3f}
    
    **User Details:**
    - User ID: {alert.get('user_id', 'Unknown')}
    - IP Address: {alert.get('ip_address', 'Unknown')}
    - Location: {alert.get('location', 'Unknown')}
    
    **Device Information:**
    - Type: {alert.get('device_info', {}).get('type', 'Unknown')}
    - Browser: {alert.get('device_info', {}).get('browser', 'Unknown')}
    - OS: {alert.get('device_info', {}).get('os', 'Unknown')}
    
    **Login Status:** {'✅ Success' if alert.get('login_success', False) else '❌ Failed'}
    """
    
    return message.strip()

def export_data_to_csv(data: pd.DataFrame, filename: str = None) -> str:
    """
    Export data to CSV format for download.
    """
    if filename is None:
        filename = f"login_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return data.to_csv(index=False)

def import_data_from_csv(uploaded_file) -> pd.DataFrame:
    """
    Import data from uploaded CSV file.
    """
    try:
        data = pd.read_csv(uploaded_file)
        logger.info(f"Successfully imported {len(data)} records from CSV")
        return data
    except Exception as e:
        logger.error(f"Error importing CSV data: {str(e)}")
        st.error(f"Error importing data: {str(e)}")
        return pd.DataFrame()

def validate_data_quality(data: pd.DataFrame) -> Dict:
    """
    Validate data quality and return quality metrics.
    """
    quality_metrics = {
        'total_records': len(data),
        'missing_values': {},
        'data_types': {},
        'duplicates': 0,
        'date_range': {},
        'quality_score': 0.0,
        'issues': []
    }
    
    if len(data) == 0:
        quality_metrics['issues'].append("No data provided")
        return quality_metrics

    missing_counts = data.isnull().sum()
    quality_metrics['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    quality_metrics['data_types'] = data.dtypes.astype(str).to_dict()

    quality_metrics['duplicates'] = data.duplicated().sum()

    if 'timestamp' in data.columns:
        try:
            timestamps = pd.to_datetime(data['timestamp'])
            quality_metrics['date_range'] = {
                'start': timestamps.min().isoformat(),
                'end': timestamps.max().isoformat(),
                'span_days': (timestamps.max() - timestamps.min()).days
            }
        except Exception as e:
            quality_metrics['issues'].append(f"Invalid timestamp format: {str(e)}")

    score = 100

    missing_ratio = sum(missing_counts) / (len(data) * len(data.columns))
    score -= missing_ratio * 30
    
    duplicate_ratio = quality_metrics['duplicates'] / len(data)
    score -= duplicate_ratio * 20

    required_columns = ['user_id', 'timestamp', 'ip_address', 'latitude', 'longitude']
    missing_required = [col for col in required_columns if col not in data.columns]
    score -= len(missing_required) * 10
    
    quality_metrics['quality_score'] = max(score, 0)
 
    if missing_required:
        quality_metrics['issues'].append(f"Missing required columns: {missing_required}")
    
    if missing_ratio > 0.1:
        quality_metrics['issues'].append(f"High missing value ratio: {missing_ratio:.2%}")
    
    if duplicate_ratio > 0.05:
        quality_metrics['issues'].append(f"High duplicate ratio: {duplicate_ratio:.2%}")
    
    return quality_metrics

def generate_security_report(data: pd.DataFrame, risk_scores: np.ndarray, 
                           alerts: List[Dict]) -> Dict:
    """
    Generate comprehensive security report.
    """
    report_timestamp = datetime.now().isoformat()

    basic_stats = calculate_risk_metrics(data, risk_scores)
 
    alert_stats = {
        'total_alerts': len(alerts),
        'high_risk_alerts': len([a for a in alerts if a.get('alert_level') == 'HIGH']),
        'medium_risk_alerts': len([a for a in alerts if a.get('alert_level') == 'MEDIUM']),
        'unique_users_alerted': len(set(a.get('user_id', '') for a in alerts)),
        'most_common_risk_factors': {}
    }
    
    risk_factor_counts = {}
    for alert in alerts:
        factors = alert.get('risk_factors', {})
        for factor, score in factors.items():
            if score > 0.3 and factor != 'total_additional_risk':
                risk_factor_counts[factor] = risk_factor_counts.get(factor, 0) + 1
    
    alert_stats['most_common_risk_factors'] = dict(
        sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    )
    
    geo_stats = {
        'unique_countries': data['country'].nunique(),
        'unique_cities': data['city'].nunique(),
        'top_countries_by_volume': data['country'].value_counts().head(5).to_dict(),
        'high_risk_locations': {}
    }

    if len(risk_scores) > 0:
        data_with_risk = data.copy()
        data_with_risk['risk_score'] = risk_scores
        high_risk_locations = data_with_risk[data_with_risk['risk_score'] > 0.8]
        
        if len(high_risk_locations) > 0:
            geo_stats['high_risk_locations'] = high_risk_locations.groupby(['country', 'city']).size().to_dict()

    device_stats = {
        'unique_devices': data['device_type'].nunique(),
        'unique_browsers': data['browser'].nunique(),
        'unique_os': data['operating_system'].nunique(),
        'device_distribution': data['device_type'].value_counts().to_dict(),
        'browser_distribution': data['browser'].value_counts().to_dict(),
        'os_distribution': data['operating_system'].value_counts().to_dict()
    }

    data['timestamp_dt'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp_dt'].dt.hour
    data['day_of_week'] = data['timestamp_dt'].dt.day_name()
    
    temporal_stats = {
        'date_range': {
            'start': data['timestamp_dt'].min().isoformat(),
            'end': data['timestamp_dt'].max().isoformat()
        },
        'peak_hour': data['hour'].value_counts().idxmax(),
        'peak_day': data['day_of_week'].value_counts().idxmax(),
        'hourly_distribution': data['hour'].value_counts().sort_index().to_dict(),
        'daily_distribution': data['day_of_week'].value_counts().to_dict()
    }

    recommendations = generate_security_recommendations(
        basic_stats, alert_stats, geo_stats, device_stats, temporal_stats
    )
    
    return {
        'report_timestamp': report_timestamp,
        'summary': {
            'total_logins_analyzed': len(data),
            'analysis_period': temporal_stats['date_range'],
            'overall_risk_level': 'HIGH' if basic_stats['avg_risk_score'] > 0.7 else 'MEDIUM' if basic_stats['avg_risk_score'] > 0.4 else 'LOW',
            'key_findings': generate_key_findings(basic_stats, alert_stats, geo_stats)
        },
        'risk_analysis': basic_stats,
        'alert_analysis': alert_stats,
        'geographic_analysis': geo_stats,
        'device_analysis': device_stats,
        'temporal_analysis': temporal_stats,
        'recommendations': recommendations
    }

def generate_key_findings(basic_stats: Dict, alert_stats: Dict, geo_stats: Dict) -> List[str]:
    """Generate key findings for the security report."""
    findings = []
    
    if basic_stats['avg_risk_score'] > 0.7:
        findings.append(f"High average risk score detected: {basic_stats['avg_risk_score']:.3f}")
    
    if alert_stats['total_alerts'] > 0:
        alert_rate = alert_stats['total_alerts'] / basic_stats['total_logins'] * 100
        findings.append(f"Alert rate: {alert_rate:.1f}% of logins triggered alerts")
    
    # High-risk findings
    if basic_stats['high_risk_count'] > 0:
        high_risk_rate = basic_stats['high_risk_count'] / basic_stats['total_logins'] * 100
        findings.append(f"{high_risk_rate:.1f}% of logins classified as high-risk")
    
    # Geographic findings
    if geo_stats['unique_countries'] > 20:
        findings.append(f"Wide geographic spread detected: {geo_stats['unique_countries']} countries")

    common_factors = alert_stats.get('most_common_risk_factors', {})
    if common_factors:
        top_factor = max(common_factors.items(), key=lambda x: x[1])
        findings.append(f"Most common risk factor: {top_factor[0]} ({top_factor[1]} occurrences)")
    
    return findings

def generate_security_recommendations(basic_stats: Dict, alert_stats: Dict, 
                                    geo_stats: Dict, device_stats: Dict, 
                                    temporal_stats: Dict) -> List[str]:
    """Generate actionable security recommendations."""
    recommendations = []
    
    # Risk-based recommendations
    if basic_stats['avg_risk_score'] > 0.6:
        recommendations.append("Consider lowering risk thresholds to catch more suspicious activities")
    
    if basic_stats['high_risk_count'] > basic_stats['total_logins'] * 0.1:
        recommendations.append("Implement additional authentication factors for high-risk logins")
    
    # Alert-based recommendations
    if alert_stats['total_alerts'] > 0:
        alert_rate = alert_stats['total_alerts'] / basic_stats['total_logins']
        if alert_rate > 0.2:
            recommendations.append("High alert volume detected - review and tune detection models")
        elif alert_rate < 0.01:
            recommendations.append("Very low alert volume - consider increasing sensitivity")
    
    # Geographic recommendations
    if geo_stats['unique_countries'] > 50:
        recommendations.append("Implement geo-blocking for high-risk countries")
    
    if geo_stats.get('high_risk_locations'):
        recommendations.append("Review and potentially block access from consistently high-risk locations")
    
    # Device recommendations
    device_diversity = len(device_stats['device_distribution'])
    if device_diversity > 10:
        recommendations.append("High device diversity detected - strengthen device fingerprinting")
    
    # Temporal recommendations
    peak_hour = temporal_stats.get('peak_hour', 12)
    if peak_hour < 6 or peak_hour > 22:
        recommendations.append(f"Unusual peak activity hour ({peak_hour}:00) - investigate off-hours access")
    
    # General recommendations
    recommendations.extend([
        "Regularly update ML models with new threat intelligence",
        "Implement user behavior analytics for improved baseline detection",
        "Consider integrating with SIEM systems for centralized monitoring",
        "Establish incident response procedures for high-risk alerts",
        "Conduct regular security awareness training for users"
    ])
    
    return recommendations

def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for privacy protection."""
    return hashlib.sha256(data.encode()).hexdigest()[:16]

def anonymize_user_data(data: pd.DataFrame) -> pd.DataFrame:
    """Anonymize user data for privacy protection."""
    anonymized = data.copy()

    if 'user_id' in anonymized.columns:
        anonymized['user_id'] = anonymized['user_id'].apply(hash_sensitive_data)

    if 'ip_address' in anonymized.columns:
        anonymized['ip_address'] = anonymized['ip_address'].apply(hash_sensitive_data)
    
    if 'device_fingerprint' in anonymized.columns:
        anonymized['device_fingerprint'] = anonymized['device_fingerprint'].apply(hash_sensitive_data)
    
    return anonymized

@st.cache_data
def cached_data_processing(data_json: str) -> Dict:
    """Cache expensive data processing operations."""
    try:
        data_dict = json.loads(data_json)
        return data_dict
    except Exception as e:
        logger.error(f"Error in cached data processing: {str(e)}")
        return {}

def format_number(num: float, precision: int = 2) -> str:
    """Format numbers for display in dashboard."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.{precision}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def get_time_ago(timestamp: str) -> str:
    """Get human-readable time difference."""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} days ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hours ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "Just now"
    except Exception:
        return "Unknown"
