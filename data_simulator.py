import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import hashlib

class LoginDataSimulator:
    def __init__(self, seed=42):
        """Initialize the login data simulator with configurable parameters."""
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Define realistic distributions
        self.device_types = ['Desktop', 'Mobile', 'Tablet']
        self.device_weights = [0.4, 0.5, 0.1]
        
        self.browsers = ['Chrome', 'Safari', 'Firefox', 'Edge', 'Opera']
        self.browser_weights = [0.6, 0.2, 0.1, 0.08, 0.02]
        
        self.operating_systems = ['Windows', 'macOS', 'Linux', 'iOS', 'Android']
        self.os_weights = [0.45, 0.25, 0.1, 0.1, 0.1]
        
        # Common cities and their coordinates for realistic geo distribution
        self.major_cities = [
            ('New York', 40.7128, -74.0060, 'United States'),
            ('London', 51.5074, -0.1278, 'United Kingdom'),
            ('Tokyo', 35.6762, 139.6503, 'Japan'),
            ('Paris', 48.8566, 2.3522, 'France'),
            ('Berlin', 52.5200, 13.4050, 'Germany'),
            ('Sydney', -33.8688, 151.2093, 'Australia'),
            ('Toronto', 43.6532, -79.3832, 'Canada'),
            ('Mumbai', 19.0760, 72.8777, 'India'),
            ('Singapore', 1.3521, 103.8198, 'Singapore'),
            ('Dubai', 25.2048, 55.2708, 'United Arab Emirates'),
            ('São Paulo', -23.5505, -46.6333, 'Brazil'),
            ('Moscow', 55.7558, 37.6173, 'Russia'),
            ('Hong Kong', 22.3193, 114.1694, 'Hong Kong'),
            ('Amsterdam', 52.3676, 4.9041, 'Netherlands'),
            ('Stockholm', 59.3293, 18.0686, 'Sweden')
        ]
        
        # Suspicious patterns for anomaly generation
        self.suspicious_countries = [
            ('Unknown Location', 0.0, 0.0, 'Unknown'),
            ('Remote Area', 45.0, 90.0, 'Suspicious'),
            ('Blocked Region', -45.0, -90.0, 'Blocked')
        ]
        
    def generate_user_profile(self, user_id):
        """Generate a consistent user profile with typical behavior patterns."""
        return {
            'user_id': user_id,
            'home_country': random.choice([city[3] for city in self.major_cities]),
            'primary_device': np.random.choice(self.device_types, p=self.device_weights),
            'primary_browser': np.random.choice(self.browsers, p=self.browser_weights),
            'primary_os': np.random.choice(self.operating_systems, p=self.os_weights),
            'typical_login_hours': sorted(random.sample(range(6, 23), k=random.randint(3, 8))),
            'work_days_only': random.choice([True, False]),
            'travel_frequency': random.uniform(0.05, 0.3) 
        }
    
    def generate_device_fingerprint(self, device_type, browser, os):
        """Generate a realistic device fingerprint hash."""
        fingerprint_data = f"{device_type}_{browser}_{os}_{random.randint(1000, 9999)}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()[:16]
    
    def is_anomalous_login(self, user_profile, login_time, location, device_info):
        """Determine if a login should be considered anomalous based on user patterns."""
        anomaly_score = 0.0
        
        # Time-based anomalies
        hour = login_time.hour
        if hour not in user_profile['typical_login_hours']:
            anomaly_score += 0.3
        
        # Weekend vs workday patterns
        is_weekend = login_time.weekday() >= 5
        if user_profile['work_days_only'] and is_weekend:
            anomaly_score += 0.2
        
        # Location anomalies
        if location[3] != user_profile['home_country']:
            if random.random() > user_profile['travel_frequency']:
                anomaly_score += 0.4
        
        # Device anomalies
        if device_info['device_type'] != user_profile['primary_device']:
            anomaly_score += 0.2
        if device_info['browser'] != user_profile['primary_browser']:
            anomaly_score += 0.1
        if device_info['operating_system'] != user_profile['primary_os']:
            anomaly_score += 0.1
        
        return anomaly_score > 0.5
    
    def generate_login_data(self, num_users=100, num_days=30, anomaly_rate=0.1):
        """Generate realistic login data with controlled anomaly injection."""
        
        user_profiles = {}
        for i in range(num_users):
            user_id = f"user_{i:04d}"
            user_profiles[user_id] = self.generate_user_profile(user_id)
        
        login_records = []
        start_date = datetime.now() - timedelta(days=num_days)
       
        total_logins = num_users * num_days * random.randint(1, 5)  # 1-5 logins per user per day on average
        
        for _ in range(total_logins):
            user_id = random.choice(list(user_profiles.keys()))
            user_profile = user_profiles[user_id]
       
            days_offset = random.uniform(0, num_days)
            login_time = start_date + timedelta(days=days_offset)
            force_anomaly = random.random() < anomaly_rate
            
            if force_anomaly:
                location = random.choice(self.suspicious_countries + self.major_cities)
                
                atypical_hours = [h for h in range(24) if h not in user_profile['typical_login_hours']]
                if atypical_hours:
                    hour = random.choice(atypical_hours)
                    login_time = login_time.replace(hour=hour, minute=random.randint(0, 59))
                
                # Anomalous device
                device_type = random.choice(self.device_types)
                browser = random.choice(self.browsers)
                operating_system = random.choice(self.operating_systems)
            else:
                location = None
                for city_info in self.major_cities:
                    if city_info[3] == user_profile['home_country']:
                        if random.random() < 0.8:  # 80% chance of home country login
                            location = city_info
                            break
                
                if location is None:
                    location = random.choice(self.major_cities)
                
                if user_profile['typical_login_hours']:
                    hour = random.choice(user_profile['typical_login_hours'])
                    login_time = login_time.replace(hour=hour, minute=random.randint(0, 59))
 
                device_type = user_profile['primary_device'] if random.random() < 0.7 else random.choice(self.device_types)
                browser = user_profile['primary_browser'] if random.random() < 0.8 else random.choice(self.browsers)
                operating_system = user_profile['primary_os'] if random.random() < 0.8 else random.choice(self.operating_systems)
            
            device_fingerprint = self.generate_device_fingerprint(device_type, browser, operating_system)
            ip_address = self.fake.ipv4()
   
            login_record = {
                'timestamp': login_time,
                'user_id': user_id,
                'ip_address': ip_address,
                'latitude': location[1],
                'longitude': location[2],
                'city': location[0],
                'country': location[3],
                'device_type': device_type,
                'browser': browser,
                'operating_system': operating_system,
                'device_fingerprint': device_fingerprint,
                'success': random.choice([True, True, True, False]),  # 75% success rate
                'session_duration_minutes': random.randint(5, 180)
            }
            
            login_records.append(login_record)

        df = pd.DataFrame(login_records)
        df = df.sort_values('timestamp').reset_index(drop=True)

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['timestamp'].dt.weekday >= 5
    
        df['prev_latitude'] = df.groupby('user_id')['latitude'].shift(1)
        df['prev_longitude'] = df.groupby('user_id')['longitude'].shift(1)
        df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
  
        df['distance_from_prev'] = np.sqrt(
            (df['latitude'] - df['prev_latitude'])**2 + 
            (df['longitude'] - df['prev_longitude'])**2
        ) * 111 

        df['time_since_prev_hours'] = (
            df['timestamp'] - df['prev_timestamp']
        ).dt.total_seconds() / 3600

        df['velocity_kmh'] = np.where(
            df['time_since_prev_hours'] > 0,
            df['distance_from_prev'] / df['time_since_prev_hours'],
            0
        )
        
        df['distance_from_prev'] = df['distance_from_prev'].fillna(0)
        df['time_since_prev_hours'] = df['time_since_prev_hours'].fillna(0)
        df['velocity_kmh'] = df['velocity_kmh'].fillna(0)
    
        df = df.drop(['prev_latitude', 'prev_longitude', 'prev_timestamp'], axis=1)
        
        return df
    
    def add_behavioral_features(self, df):
        """Add advanced behavioral features to the dataset."""
 
        user_stats = df.groupby('user_id').agg({
            'hour': lambda x: list(x),
            'day_of_week': lambda x: list(x),
            'country': lambda x: list(x),
            'device_type': lambda x: list(x),
            'browser': lambda x: list(x)
        }).reset_index()
        
        user_stats['hour_variance'] = user_stats['hour'].apply(np.var)
        user_stats['unique_countries'] = user_stats['country'].apply(lambda x: len(set(x)))
        user_stats['unique_devices'] = user_stats['device_type'].apply(lambda x: len(set(x)))
        
        # Merge back to main dataframe
        df = df.merge(user_stats[['user_id', 'hour_variance', 'unique_countries', 'unique_devices']], on='user_id')
        
        return df
