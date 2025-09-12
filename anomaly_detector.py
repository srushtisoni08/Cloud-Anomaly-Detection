import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

class RealTimeAnomalyDetector:
    """
    Real-time anomaly detection system for login events.
    Processes individual login events and generates alerts based on risk scores.
    """
    
    def __init__(self, ml_models, high_risk_threshold=0.8, medium_risk_threshold=0.5):
        """
        Initialize the real-time anomaly detector.
        
        Args:
            ml_models: Trained ML models instance
            high_risk_threshold: Threshold for high-risk alerts (default: 0.8)
            medium_risk_threshold: Threshold for medium-risk alerts (default: 0.5)
        """
        self.ml_models = ml_models
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        
        # Alert tracking
        self.alert_history = []
        self.user_baselines = {}
        self.blocked_ips = set()
        self.suspicious_patterns = []
        
        # Rate limiting tracking
        self.login_attempts = {}  # IP -> [(timestamp, success), ...]
        self.max_attempts_per_minute = 10
        self.max_attempts_per_hour = 50
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def process_login_event(self, login_event: Dict) -> Dict:
        """
        Process a single login event and return risk assessment.
        
        Args:
            login_event: Dictionary containing login event data
            
        Returns:
            Dictionary with risk assessment and recommended actions
        """
        try:
            # Convert to DataFrame for ML model compatibility
            event_df = pd.DataFrame([login_event])
            
            # Calculate risk score using ML models
            risk_score = self.ml_models.calculate_risk_scores(event_df)[0]
            
            # Perform additional real-time checks
            additional_risk_factors = self._perform_realtime_checks(login_event)
            
            # Combine ML risk score with real-time factors
            combined_risk = min(risk_score + additional_risk_factors['total_additional_risk'], 1.0)
            
            # Determine alert level
            alert_level = self._determine_alert_level(combined_risk)
            
            # Generate recommended actions
            recommended_actions = self._generate_recommendations(
                login_event, combined_risk, alert_level, additional_risk_factors
            )
            
            # Create alert if necessary
            alert = None
            if alert_level in ['HIGH', 'MEDIUM']:
                alert = self._create_alert(login_event, combined_risk, alert_level, additional_risk_factors)
                self.alert_history.append(alert)
            
            # Update user baseline
            self._update_user_baseline(login_event)
            
            # Track login attempts for rate limiting
            self._track_login_attempt(login_event)
            
            return {
                'risk_score': float(combined_risk),
                'alert_level': alert_level,
                'ml_risk_score': float(risk_score),
                'additional_risk_factors': additional_risk_factors,
                'recommended_actions': recommended_actions,
                'alert': alert,
                'timestamp_processed': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing login event: {str(e)}")
            return {
                'error': str(e),
                'risk_score': 1.0,  # Default to high risk on error
                'alert_level': 'HIGH',
                'recommended_actions': ['BLOCK_LOGIN', 'MANUAL_REVIEW']
            }
    
    def _perform_realtime_checks(self, login_event: Dict) -> Dict:
        """Perform real-time security checks beyond ML models."""
        risk_factors = {
            'rate_limiting': 0.0,
            'impossible_travel': 0.0,
            'suspicious_ip': 0.0,
            'device_reputation': 0.0,
            'time_anomaly': 0.0,
            'geolocation_risk': 0.0,
            'total_additional_risk': 0.0
        }
        
        # 1. Rate limiting check
        rate_limit_risk = self._check_rate_limiting(login_event)
        risk_factors['rate_limiting'] = rate_limit_risk
        
        # 2. Impossible travel detection
        travel_risk = self._check_impossible_travel(login_event)
        risk_factors['impossible_travel'] = travel_risk
        
        # 3. IP reputation check
        ip_risk = self._check_ip_reputation(login_event)
        risk_factors['suspicious_ip'] = ip_risk
        
        # 4. Device reputation
        device_risk = self._check_device_reputation(login_event)
        risk_factors['device_reputation'] = device_risk
        
        # 5. Time-based anomalies
        time_risk = self._check_time_anomalies(login_event)
        risk_factors['time_anomaly'] = time_risk
        
        # 6. Geolocation risks
        geo_risk = self._check_geolocation_risks(login_event)
        risk_factors['geolocation_risk'] = geo_risk
        
        # Calculate total additional risk (weighted average)
        weights = {
            'rate_limiting': 0.3,
            'impossible_travel': 0.25,
            'suspicious_ip': 0.2,
            'device_reputation': 0.1,
            'time_anomaly': 0.1,
            'geolocation_risk': 0.05
        }
        
        total_risk = sum(risk_factors[factor] * weights[factor] 
                        for factor in weights.keys())
        risk_factors['total_additional_risk'] = total_risk
        
        return risk_factors
    
    def _check_rate_limiting(self, login_event: Dict) -> float:
        """Check for suspicious login attempt rates."""
        ip = login_event.get('ip_address', '')
        current_time = datetime.now()
        
        # Get recent attempts for this IP
        if ip in self.login_attempts:
            attempts = self.login_attempts[ip]
            
            # Count attempts in last minute
            minute_ago = current_time - timedelta(minutes=1)
            recent_attempts = [a for a in attempts if a[0] > minute_ago]
            
            if len(recent_attempts) > self.max_attempts_per_minute:
                return 0.8  # Very high risk for rate limiting
            elif len(recent_attempts) > self.max_attempts_per_minute * 0.7:
                return 0.5  # Medium risk approaching limit
        
        return 0.0
    
    def _check_impossible_travel(self, login_event: Dict) -> float:
        """Check for geographically impossible travel patterns."""
        user_id = login_event.get('user_id', '')
        current_lat = login_event.get('latitude', 0)
        current_lon = login_event.get('longitude', 0)
        current_time = datetime.strptime(
            login_event.get('timestamp', datetime.now().isoformat())[:19], 
            '%Y-%m-%d %H:%M:%S'
        )
        
        if user_id in self.user_baselines:
            last_login = self.user_baselines[user_id].get('last_location')
            if last_login:
                # Calculate distance and time difference
                distance = self._calculate_distance(
                    last_login['lat'], last_login['lon'],
                    current_lat, current_lon
                )
                
                time_diff = (current_time - last_login['timestamp']).total_seconds() / 3600  # hours
                
                if time_diff > 0:
                    max_speed = distance / time_diff  # km/h
                    
                    # Commercial aircraft speed ~900 km/h
                    if max_speed > 1000:  # Impossible even by fastest commercial flight
                        return 0.9
                    elif max_speed > 500:  # Very fast, likely requires air travel
                        return 0.6
                    elif max_speed > 200:  # Fast travel, possible but unusual
                        return 0.3
        
        return 0.0
    
    def _check_ip_reputation(self, login_event: Dict) -> float:
        """Check IP address reputation and characteristics."""
        ip = login_event.get('ip_address', '')
        
        # Check against blocked IP list
        if ip in self.blocked_ips:
            return 1.0
        
        # Simple heuristics for suspicious IPs
        # In production, you'd integrate with threat intelligence feeds
        
        # Check for private/internal IPs in unexpected contexts
        if ip.startswith(('10.', '192.168.', '172.')):
            # Private IP from external login could be suspicious
            return 0.3
        
        # Check for known suspicious patterns
        suspicious_patterns = ['tor', 'proxy', 'vpn']  # Simplified check
        country = login_event.get('country', '').lower()
        
        for pattern in suspicious_patterns:
            if pattern in country:
                return 0.4
        
        return 0.0
    
    def _check_device_reputation(self, login_event: Dict) -> float:
        """Check device fingerprint and reputation."""
        device_fingerprint = login_event.get('device_fingerprint', '')
        user_id = login_event.get('user_id', '')
        
        if user_id in self.user_baselines:
            known_devices = self.user_baselines[user_id].get('known_devices', set())
            
            if device_fingerprint and device_fingerprint not in known_devices:
                # New device for user
                if len(known_devices) == 0:
                    return 0.2  # First-time user, moderate risk
                else:
                    return 0.4  # Established user with new device
        
        return 0.0
    
    def _check_time_anomalies(self, login_event: Dict) -> float:
        """Check for unusual login times for the user."""
        user_id = login_event.get('user_id', '')
        login_hour = int(login_event.get('hour', 12))
        
        if user_id in self.user_baselines:
            typical_hours = self.user_baselines[user_id].get('typical_hours', set())
            
            if typical_hours:
                if login_hour not in typical_hours:
                    # Outside typical hours
                    if login_hour < 6 or login_hour > 22:
                        return 0.5  # Late night/early morning
                    else:
                        return 0.2  # Different but reasonable hours
        
        return 0.0
    
    def _check_geolocation_risks(self, login_event: Dict) -> float:
        """Check for high-risk geographic locations."""
        country = login_event.get('country', '')
        city = login_event.get('city', '')
        
        # Simplified risk assessment based on location
        high_risk_indicators = [
            'unknown', 'suspicious', 'blocked', 'tor',
            'anonymous', 'proxy'
        ]
        
        location_text = f"{country} {city}".lower()
        
        for indicator in high_risk_indicators:
            if indicator in location_text:
                return 0.6
        
        return 0.0
    
    def _determine_alert_level(self, risk_score: float) -> str:
        """Determine alert level based on risk score."""
        if risk_score >= self.high_risk_threshold:
            return 'HIGH'
        elif risk_score >= self.medium_risk_threshold:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_recommendations(self, login_event: Dict, risk_score: float, 
                                alert_level: str, risk_factors: Dict) -> List[str]:
        """Generate recommended security actions."""
        recommendations = []
        
        if alert_level == 'HIGH':
            recommendations.extend(['BLOCK_LOGIN', 'REQUIRE_MFA', 'NOTIFY_ADMIN'])
            
            if risk_factors['rate_limiting'] > 0.5:
                recommendations.append('BLOCK_IP_TEMPORARILY')
            
            if risk_factors['impossible_travel'] > 0.8:
                recommendations.append('FREEZE_ACCOUNT_TEMPORARILY')
                
        elif alert_level == 'MEDIUM':
            recommendations.extend(['REQUIRE_MFA', 'LOG_DETAILED'])
            
            if risk_factors['device_reputation'] > 0.3:
                recommendations.append('DEVICE_VERIFICATION')
            
            if risk_factors['geolocation_risk'] > 0.3:
                recommendations.append('LOCATION_VERIFICATION')
        
        else:  # LOW risk
            recommendations.append('ALLOW_LOGIN')
        
        # Additional context-specific recommendations
        if not login_event.get('success', True):
            recommendations.append('MONITOR_FAILED_ATTEMPTS')
        
        return list(set(recommendations))  # Remove duplicates
    
    def _create_alert(self, login_event: Dict, risk_score: float, 
                     alert_level: str, risk_factors: Dict) -> Dict:
        """Create a security alert record."""
        return {
            'alert_id': f"alert_{len(self.alert_history) + 1:06d}",
            'timestamp': datetime.now().isoformat(),
            'alert_level': alert_level,
            'risk_score': float(risk_score),
            'user_id': login_event.get('user_id', 'unknown'),
            'ip_address': login_event.get('ip_address', 'unknown'),
            'location': f"{login_event.get('city', 'Unknown')}, {login_event.get('country', 'Unknown')}",
            'device_info': {
                'type': login_event.get('device_type', 'unknown'),
                'browser': login_event.get('browser', 'unknown'),
                'os': login_event.get('operating_system', 'unknown')
            },
            'risk_factors': risk_factors,
            'login_success': login_event.get('success', False),
            'raw_event': login_event
        }
    
    def _update_user_baseline(self, login_event: Dict):
        """Update user behavioral baseline with new login data."""
        user_id = login_event.get('user_id', '')
        current_time = datetime.strptime(
            login_event.get('timestamp', datetime.now().isoformat())[:19], 
            '%Y-%m-%d %H:%M:%S'
        )
        
        if user_id not in self.user_baselines:
            self.user_baselines[user_id] = {
                'typical_hours': set(),
                'known_devices': set(),
                'known_countries': set(),
                'login_count': 0,
                'last_location': None
            }
        
        baseline = self.user_baselines[user_id]
        
        # Update typical login hours
        baseline['typical_hours'].add(int(login_event.get('hour', 12)))
        
        # Update known devices
        device_fp = login_event.get('device_fingerprint', '')
        if device_fp:
            baseline['known_devices'].add(device_fp)
        
        # Update known countries
        baseline['known_countries'].add(login_event.get('country', ''))
        
        # Update location tracking
        baseline['last_location'] = {
            'lat': login_event.get('latitude', 0),
            'lon': login_event.get('longitude', 0),
            'timestamp': current_time
        }
        
        # Update login count
        baseline['login_count'] += 1
        
        # Limit baseline size to prevent memory issues
        if len(baseline['typical_hours']) > 12:
            # Keep most common hours
            hour_counts = {}
            # This is simplified - in production, you'd track actual frequencies
            baseline['typical_hours'] = set(list(baseline['typical_hours'])[:12])
        
        if len(baseline['known_devices']) > 10:
            # Keep most recent devices
            baseline['known_devices'] = set(list(baseline['known_devices'])[-10:])
    
    def _track_login_attempt(self, login_event: Dict):
        """Track login attempts for rate limiting."""
        ip = login_event.get('ip_address', '')
        success = login_event.get('success', False)
        current_time = datetime.now()
        
        if ip not in self.login_attempts:
            self.login_attempts[ip] = []
        
        self.login_attempts[ip].append((current_time, success))
        
        # Clean old attempts (keep only last hour)
        hour_ago = current_time - timedelta(hours=1)
        self.login_attempts[ip] = [
            attempt for attempt in self.login_attempts[ip] 
            if attempt[0] > hour_ago
        ]
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates using Haversine formula."""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of alerts in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        return {
            'total_alerts': len(recent_alerts),
            'high_risk_alerts': len([a for a in recent_alerts if a['alert_level'] == 'HIGH']),
            'medium_risk_alerts': len([a for a in recent_alerts if a['alert_level'] == 'MEDIUM']),
            'unique_users_affected': len(set(a['user_id'] for a in recent_alerts)),
            'top_risk_factors': self._get_top_risk_factors(recent_alerts),
            'alert_timeline': recent_alerts[-10:] if recent_alerts else []
        }
    
    def _get_top_risk_factors(self, alerts: List[Dict]) -> Dict:
        """Identify most common risk factors in recent alerts."""
        factor_counts = {}
        
        for alert in alerts:
            risk_factors = alert.get('risk_factors', {})
            for factor, score in risk_factors.items():
                if score > 0.3:  # Significant risk factors only
                    factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        return dict(sorted(factor_counts.items(), key=lambda x: x[1], reverse=True))
