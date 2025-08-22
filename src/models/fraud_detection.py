"""Anomaly detection models for fraud detection."""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Handles anomaly detection for transaction fraud."""
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Prepare features for anomaly detection."""
        
        # Select available feature columns
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        if len(available_features) == 0:
            raise ValueError("No feature columns available")
        
        # Extract features and handle missing values
        X = df[available_features].copy()
        X = X.fillna(X.median())
        
        # Replace infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X.values
    
    def train_isolation_forest(self, X: np.ndarray) -> IsolationForest:
        """Train Isolation Forest model."""
        
        logger.info("Training Isolation Forest...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model with optimized parameters
        model = IsolationForest(
            contamination=self.contamination,
            n_estimators=200,
            max_samples=0.8,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_scaled)
        self.models['isolation_forest'] = model
        self.isolation_forest = model
        
        logger.info("Isolation Forest training completed")
        return model
    
    def train_lof(self, X: np.ndarray) -> LocalOutlierFactor:
        """Train Local Outlier Factor model."""
        
        logger.info("Training Local Outlier Factor...")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Train model
        model = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            n_jobs=-1
        )
        
        # LOF doesn't have separate fit/predict, so we store the fitted model
        outlier_labels = model.fit_predict(X_scaled)
        self.models['lof'] = model
        
        logger.info("LOF training completed")
        return model
    
    def train_autoencoder(self, X: np.ndarray) -> Dict:
        """Train autoencoder for anomaly detection."""
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            logger.info("Training Autoencoder...")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X_scaled)
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            # Define autoencoder
            input_dim = X_scaled.shape[1]
            encoding_dim = min(32, input_dim // 2)
            
            class Autoencoder(nn.Module):
                def __init__(self, input_dim, encoding_dim):
                    super(Autoencoder, self).__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, encoding_dim * 2),
                        nn.ReLU(),
                        nn.Linear(encoding_dim * 2, encoding_dim),
                        nn.ReLU()
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(encoding_dim, encoding_dim * 2),
                        nn.ReLU(),
                        nn.Linear(encoding_dim * 2, input_dim)
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded
            
            model = Autoencoder(input_dim, encoding_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train
            model.train()
            for epoch in range(100):
                total_loss = 0
                for batch_x, _ in dataloader:
                    optimizer.zero_grad()
                    reconstructed = model(batch_x)
                    loss = criterion(reconstructed, batch_x)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
            
            self.models['autoencoder'] = {
                'model': model,
                'input_dim': input_dim,
                'encoding_dim': encoding_dim
            }
            
            logger.info("Autoencoder training completed")
            return self.models['autoencoder']
            
        except ImportError:
            logger.warning("PyTorch not available, skipping autoencoder training")
            return None
    
    def predict_anomalies(self, X: np.ndarray, model_type: str = 'isolation_forest') -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using specified model."""
        
        if model_type == 'isolation_forest' and self.isolation_forest is None:
            raise ValueError(f"Model {model_type} not trained")
        elif model_type == 'lof' and self.lof is None:
            raise ValueError(f"Model {model_type} not trained")
        elif model_type == 'autoencoder' and model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained")
        
        X_scaled = self.scaler.transform(X)
        
        if model_type == 'isolation_forest':
            model = self.isolation_forest
            anomaly_scores = model.decision_function(X_scaled)
            predictions = model.predict(X_scaled)
            predictions = (predictions == -1).astype(int)  # Convert to 0/1
            
        elif model_type == 'lof':
            # LOF requires retraining for new data
            model = LocalOutlierFactor(n_neighbors=20, contamination=self.contamination)
            predictions = model.fit_predict(X_scaled)
            anomaly_scores = model.negative_outlier_factor_
            predictions = (predictions == -1).astype(int)
            
        elif model_type == 'autoencoder':
            try:
                import torch
                model_dict = self.models['autoencoder']
                model = model_dict['model']
                model.eval()
                
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled)
                    reconstructed = model(X_tensor)
                    mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
                    anomaly_scores = mse.numpy()
                
                # Convert scores to binary predictions using threshold
                threshold = np.percentile(anomaly_scores, (1 - self.contamination) * 100)
                predictions = (anomaly_scores > threshold).astype(int)
                
            except ImportError:
                raise ValueError("PyTorch not available for autoencoder prediction")
        
        return predictions, anomaly_scores
    
    def load_models(self):
        """Load trained models from disk."""
        try:
            if os.path.exists('models/isolation_forest.pkl'):
                self.isolation_forest = joblib.load('models/isolation_forest.pkl')
                print(f"Loaded isolation_forest model")
                
            if os.path.exists('models/lof.pkl'):
                self.lof = joblib.load('models/lof.pkl')
                print(f"Loaded LOF model")
                
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
                
            if os.path.exists('models/feature_columns.pkl'):
                self.feature_columns = joblib.load('models/feature_columns.pkl')
                
            logger.info("Models loaded from models/")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_feature_columns(self):
        """Get the feature columns used for training."""
        if hasattr(self, 'feature_columns'):
            return self.feature_columns
        else:
            # Return default feature columns if not set
            return ['amount', 'hour', 'day_of_week', 'month', 'amount_zscore', 
                   'merchant_freq', 'category_encoded']
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Evaluate anomaly detection model."""
        
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'alert_rate': y_pred.mean(),
            'fraud_rate': y_true.mean()
        }
        
        # AUC if we have scores
        if len(np.unique(y_scores)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_scores)
        
        return metrics
    
    def save_models(self, model_path: str = "models/"):
        """Save trained models."""
        
        import os
        os.makedirs(model_path, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, f"{model_path}/scaler.pkl")
        
        # Save sklearn models
        for model_name in ['isolation_forest']:
            if model_name in self.models:
                joblib.dump(self.models[model_name], f"{model_path}/{model_name}.pkl")
        
        # Save autoencoder separately if available
        if 'autoencoder' in self.models:
            try:
                import torch
                torch.save(
                    self.models['autoencoder']['model'].state_dict(),
                    f"{model_path}/autoencoder.pth"
                )
            except ImportError:
                pass
        

def main():
    """Main function for training anomaly detection models."""

    logging.basicConfig(level=logging.INFO)

    
    from sqlalchemy import create_engine
    
    # Load feature data
    engine = create_engine("sqlite:///data/transactions.db")
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM feature_store", conn)
        
        if len(df) == 0:
            print("No feature data found. Run feature engineering first.")
            return
        
        # Initialize components
        detector = AnomalyDetector()
        
        # Get feature columns (exclude non-numeric and ID columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['transaction_id', 'account_id', 'timestamp', 'is_fraud', 'unix_timestamp']
        feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"Using {len(feature_columns)} features for anomaly detection")
        
        # Prepare features and train
        X = detector.prepare_features(df, feature_columns)
        detector.train_isolation_forest(X)
        detector.train_lof(X)
        detector.train_autoencoder(X)
        
        # Evaluate on training data (for demonstration)
        if 'is_fraud' in df.columns:
            y_true = df['is_fraud'].values
            
            for model_type in ['isolation_forest']:
                if model_type in detector.models:
                    y_pred, y_scores = detector.predict_anomalies(X, model_type)
                    metrics = detector.evaluate_model(y_true, y_pred, y_scores)
                    
                    print(f"\n{model_type.upper()} Results:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.3f}")
        
        # Save models
        detector.save_models()
        print("\nModels saved successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run data loading and feature engineering first.")

if __name__ == "__main__":
    main()
