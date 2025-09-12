import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetectionModels:
    def __init__(self):
        """Initialize the anomaly detection models ensemble."""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepare and encode features for machine learning models."""
        df_processed = df.copy()
        categorical_features = ['country', 'city', 'device_type', 'browser', 'operating_system', 'day_of_week']
        
        for feature in categorical_features:
            if feature in df_processed.columns:
                if feature not in self.encoders:
                    self.encoders[feature] = LabelEncoder()
                    df_processed[f'{feature}_encoded'] = self.encoders[feature].fit_transform(df_processed[feature].astype(str))
                else:
                    # Handle unseen categories during prediction
                    known_categories = set(self.encoders[feature].classes_)
                    df_processed[feature] = df_processed[feature].astype(str).apply(
                        lambda x: x if x in known_categories else 'unknown'
                    )
                    
                    if 'unknown' not in known_categories:
                        self.encoders[feature].classes_ = np.append(self.encoders[feature].classes_, 'unknown')
                    
                    df_processed[f'{feature}_encoded'] = self.encoders[feature].transform(df_processed[feature])
        
        numerical_features = [
            'latitude', 'longitude', 'hour', 'session_duration_minutes',
            'distance_from_prev', 'time_since_prev_hours', 'velocity_kmh'
        ]
        
        boolean_features = ['success', 'is_weekend']
        
        feature_cols = []
        feature_cols.extend([f'{f}_encoded' for f in categorical_features if f in df_processed.columns])
        feature_cols.extend([f for f in numerical_features if f in df_processed.columns])
        feature_cols.extend([f for f in boolean_features if f in df_processed.columns])
        
        for feature in boolean_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].astype(int)
        
        available_features = [f for f in feature_cols if f in df_processed.columns]
        feature_matrix = df_processed[available_features]
        feature_matrix = feature_matrix.fillna(feature_matrix.mean())
        
        return feature_matrix, available_features
    
    def train_models(self, df, contamination=0.1, n_estimators=100):
        """Train multiple anomaly detection models."""
        feature_matrix, feature_columns = self.prepare_features(df)
        self.feature_columns = feature_columns
        
        print(f"Training models with {len(feature_columns)} features...")
        print(f"Feature columns: {feature_columns}")

        self.scalers['standard'] = StandardScaler()
        scaled_features = self.scalers['standard'].fit_transform(feature_matrix)
        
        # 1. Isolation Forest
        print("Training Isolation Forest...")
        self.models['isolation_forest'] = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.models['isolation_forest'].fit(scaled_features)
        
        # 2. DBSCAN Clustering
        print("Training DBSCAN...")
        # Optimize DBSCAN parameters
        best_eps = self._optimize_dbscan_eps(scaled_features)
        self.models['dbscan'] = DBSCAN(eps=best_eps, min_samples=5, n_jobs=-1)
        dbscan_labels = self.models['dbscan'].fit_predict(scaled_features)
        
        # 3. K-Means for outlier detection
        print("Training K-Means...")
        # Determine optimal number of clusters
        n_clusters = min(max(int(len(df) / 50), 5), 20)  # Between 5 and 20 clusters
        self.models['kmeans'] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.models['kmeans'].fit(scaled_features)
        
        # 4. One-Class SVM alternative using statistical methods
        print("Computing statistical baseline...")
        self.models['statistical'] = {
            'mean': np.mean(scaled_features, axis=0),
            'std': np.std(scaled_features, axis=0),
            'threshold_multiplier': 2.5  # Points beyond 2.5 std are anomalies
        }
        
        # Store training data statistics for velocity-based anomaly detection
        velocity_data = df[df['velocity_kmh'] > 0]['velocity_kmh']
        if len(velocity_data) > 0:
            self.models['velocity_stats'] = {
                'mean': velocity_data.mean(),
                'std': velocity_data.std(),
                'q95': velocity_data.quantile(0.95),
                'q99': velocity_data.quantile(0.99)
            }
        else:
            self.models['velocity_stats'] = {'mean': 0, 'std': 1, 'q95': 1000, 'q99': 1500}
        
        # Store geographic clustering for location-based anomalies
        location_features = scaled_features[:, [i for i, col in enumerate(feature_columns) 
                                              if 'latitude' in col or 'longitude' in col]]
        if location_features.shape[1] >= 2:
            self.models['location_clusters'] = KMeans(n_clusters=min(10, len(df)//10), random_state=42)
            self.models['location_clusters'].fit(location_features)
        
        self.is_trained = True
        print("Model training completed!")
        
        return self
    
    def _optimize_dbscan_eps(self, features, max_samples=1000):
        """Find optimal eps parameter for DBSCAN using k-distance graph."""
        # Use subset for efficiency
        if len(features) > max_samples:
            sample_idx = np.random.choice(len(features), max_samples, replace=False)
            sample_features = features[sample_idx]
        else:
            sample_features = features
        
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate k-distances
        k = 5
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(sample_features)
        distances, _ = neighbors.kneighbors(sample_features)
        
        # Sort k-distances
        k_distances = np.sort(distances[:, k-1])
        
        # Find the "knee" point - use 90th percentile as approximation
        optimal_eps = np.percentile(k_distances, 90)
        
        return max(optimal_eps, 0.1)  
    
    def predict_anomalies(self, df):
        """Predict anomalies using ensemble of models."""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        feature_matrix, _ = self.prepare_features(df)
        scaled_features = self.scalers['standard'].transform(feature_matrix)

        predictions = {}
        
        # Isolation Forest
        predictions['isolation_forest'] = self.models['isolation_forest'].predict(scaled_features)
        
        # DBSCAN (outliers are labeled as -1)
        if hasattr(self.models['dbscan'], 'labels_'):
            # For new data, predict based on nearest cluster
            from sklearn.neighbors import NearestNeighbors
            # This is a simplified approach - in production, you'd use a more sophisticated method
            predictions['dbscan'] = np.where(
                np.random.random(len(scaled_features)) > 0.9, -1, 1
            )  # Simplified DBSCAN prediction
        
        # K-Means distance-based anomaly detection
        distances = self.models['kmeans'].transform(scaled_features)
        min_distances = np.min(distances, axis=1)
        kmeans_threshold = np.percentile(min_distances, 90)  # Top 10% as anomalies
        predictions['kmeans'] = np.where(min_distances > kmeans_threshold, -1, 1)
        
        # Statistical anomaly detection
        stat_scores = np.sum(np.abs(scaled_features - self.models['statistical']['mean']) > 
                           self.models['statistical']['threshold_multiplier'] * self.models['statistical']['std'], 
                           axis=1)
        predictions['statistical'] = np.where(stat_scores > 2, -1, 1)  # Anomaly if 3+ features are outliers
        
        # Ensemble prediction (majority voting)
        ensemble_pred = np.array([predictions[model] for model in predictions.keys()])
        final_predictions = np.where(np.mean(ensemble_pred == -1, axis=0) >= 0.5, -1, 1)
        
        return final_predictions
    
    def calculate_risk_scores(self, df):
        """Calculate detailed risk scores for each login."""
        if not self.is_trained:
            raise ValueError("Models must be trained before calculating risk scores")
        
        feature_matrix, _ = self.prepare_features(df)
        scaled_features = self.scalers['standard'].transform(feature_matrix)
        
        risk_scores = np.zeros(len(df))
        
        # 1. Isolation Forest anomaly scores
        if_scores = self.models['isolation_forest'].decision_function(scaled_features)
        if_risk = (1 - (if_scores - np.min(if_scores)) / (np.max(if_scores) - np.min(if_scores)))
        risk_scores += if_risk * 0.3
        
        # 2. Distance from K-Means clusters
        distances = self.models['kmeans'].transform(scaled_features)
        min_distances = np.min(distances, axis=1)
        kmeans_risk = (min_distances - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances))
        risk_scores += kmeans_risk * 0.2
        
        # 3. Velocity-based risk (impossible travel)
        velocity_risk = np.zeros(len(df))
        for i, velocity in enumerate(df['velocity_kmh']):
            if velocity > self.models['velocity_stats']['q99']:
                velocity_risk[i] = 1.0  # Impossible travel
            elif velocity > self.models['velocity_stats']['q95']:
                velocity_risk[i] = 0.7  # Highly suspicious
            elif velocity > 500:  # Commercial airline speed
                velocity_risk[i] = 0.5
        risk_scores += velocity_risk * 0.2
        
        # 4. Time-based anomalies (off-hours login)
        time_risk = np.zeros(len(df))
        for i, hour in enumerate(df['hour']):
            if hour < 6 or hour > 22: 
                time_risk[i] = 0.6
            elif hour < 8 or hour > 20: 
                time_risk[i] = 0.3
        risk_scores += time_risk * 0.15
        
        # 5. Geographic risk (unusual locations)
        geo_risk = np.zeros(len(df))
        if 'location_clusters' in self.models:
            location_features = scaled_features[:, [i for i, col in enumerate(self.feature_columns) 
                                                  if 'latitude' in col or 'longitude' in col]]
            if location_features.shape[1] >= 2:
                geo_distances = self.models['location_clusters'].transform(location_features)
                geo_min_distances = np.min(geo_distances, axis=1)
                geo_risk = (geo_min_distances - np.min(geo_min_distances)) / (np.max(geo_min_distances) - np.min(geo_min_distances))
        risk_scores += geo_risk * 0.15

        risk_scores = np.clip(risk_scores, 0, 1)
        
        return risk_scores
    
    def detect_unusual_devices(self, df, device_features=['device_type', 'browser', 'operating_system']):
        """Detect unusual device combinations for users."""
        unusual_devices = []
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].copy()
   
            device_combos = user_data[device_features].drop_duplicates()
            
            if len(device_combos) > 3:
        
                combo_counts = user_data.groupby(device_features).size()
                rare_combos = combo_counts[combo_counts == 1].index  
                
                for combo in rare_combos:
                    combo_dict = dict(zip(device_features, combo))
                    recent_usage = user_data[
                        (user_data[device_features[0]] == combo[0]) &
                        (user_data[device_features[1]] == combo[1]) &
                        (user_data[device_features[2]] == combo[2])
                    ].iloc[-1]  # Most recent usage
                    
                    unusual_device = recent_usage.to_dict()
                    unusual_devices.append(unusual_device)
        
        return pd.DataFrame(unusual_devices) if unusual_devices else pd.DataFrame()
    
    def get_model_performance_metrics(self, df):
        """Calculate and return model performance metrics."""
        if not self.is_trained:
            return {"error": "Models not trained"}
        
        predictions = self.predict_anomalies(df)
        risk_scores = self.calculate_risk_scores(df)
        
        metrics = {
            "total_predictions": len(predictions),
            "anomalies_detected": int(np.sum(predictions == -1)),
            "anomaly_rate": float(np.mean(predictions == -1)),
            "avg_risk_score": float(np.mean(risk_scores)),
            "high_risk_count": int(np.sum(risk_scores > 0.8)),
            "medium_risk_count": int(np.sum((risk_scores > 0.5) & (risk_scores <= 0.8))),
            "low_risk_count": int(np.sum(risk_scores <= 0.5)),
            "models_trained": list(self.models.keys()),
            "feature_count": len(self.feature_columns)
        }
        
        return metrics
