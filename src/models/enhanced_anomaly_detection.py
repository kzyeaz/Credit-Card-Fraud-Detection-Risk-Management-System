"""
Enhanced Anomaly Detection with Ensemble Methods and Class Balancing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedAnomalyDetector:
    """Enhanced anomaly detection with ensemble methods and class balancing."""
    
    def __init__(self, contamination: float = 0.02):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_columns = []
        self.feature_importance = {}
        
        # Initialize models
        self.isolation_forest = None
        self.random_forest = None
        self.ensemble_weights = {'isolation_forest': 0.3, 'random_forest': 0.7}  # Give more weight to supervised model
        
    def prepare_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Prepare features for model training/prediction with enhanced feature engineering."""
        
        # Select only numeric features for modeling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [col for col in feature_columns if col in numeric_cols and col in df.columns]
        
        if not available_features:
            # Fallback to basic numeric features
            exclude_cols = ['transaction_id', 'account_id', 'unix_timestamp', 'is_fraud']
            available_features = [col for col in numeric_cols if col not in exclude_cols]
        
        X = df[available_features].copy()
        
        # Enhanced feature engineering for better fraud detection
        if 'amount' in X.columns:
            # Log transform for amount to handle skewness
            X['amount_log'] = np.log1p(X['amount'])
            # Amount percentile within account
            if 'account_id' in df.columns:
                X['amount_percentile'] = df.groupby('account_id')['amount'].rank(pct=True)
        
        # Time-based features if available
        if 'hour_of_day' in X.columns:
            # Cyclical encoding for hour
            X['hour_sin'] = np.sin(2 * np.pi * X['hour_of_day'] / 24)
            X['hour_cos'] = np.cos(2 * np.pi * X['hour_of_day'] / 24)
        
        if 'day_of_week' in X.columns:
            # Cyclical encoding for day of week
            X['dow_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
            X['dow_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
        
        # Handle missing values with more sophisticated imputation
        X = X.fillna(X.median())
        
        # Convert to numpy array
        X_array = X.values
        
        # Remove low variance features with stricter threshold
        if not hasattr(self, 'variance_selector'):
            self.variance_selector = VarianceThreshold(threshold=0.05)  # Stricter threshold
            X_array = self.variance_selector.fit_transform(X_array)
            self.selected_features = [X.columns[i] for i in range(len(X.columns)) 
                                    if self.variance_selector.get_support()[i]]
            logger.info(f"Removing {len(X.columns) - len(self.selected_features)} low variance features")
        else:
            X_array = self.variance_selector.transform(X_array)
        
        # Store feature columns for dashboard use
        self.feature_columns = list(X.columns)
        
        return X_array
    
    def train_models(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Dict:
        """Train ensemble of models with class balancing."""
        
        logger.info("Training enhanced anomaly detection models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest (unsupervised)
        self.isolation_forest = self._train_isolation_forest(X_scaled)
        
        # Train Random Forest (supervised) if labels available
        if y is not None:
            # Ensure y has same length as X_scaled
            if len(y) != len(X_scaled):
                logger.warning(f"Label length mismatch: X={len(X_scaled)}, y={len(y)}. Truncating to match.")
                min_len = min(len(X_scaled), len(y))
                X_scaled = X_scaled[:min_len]
                y = y[:min_len]
            
            self.random_forest = self._train_random_forest(X_scaled, y)
        
        # Store models
        self.models['isolation_forest'] = self.isolation_forest
        if self.random_forest is not None:
            self.models['random_forest'] = self.random_forest
        
        logger.info("Enhanced model training completed")
        return self.models
    
    def _train_isolation_forest(self, X: np.ndarray) -> IsolationForest:
        """Train optimized Isolation Forest."""
        
        model = IsolationForest(
            contamination=0.05,  # Reduce contamination for better precision
            n_estimators=1000,   # More estimators for stability
            max_samples=0.6,     # Smaller samples for better anomaly detection
            max_features=0.7,    # Reduce features to prevent overfitting
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X)
        logger.info("Isolation Forest training completed")
        return model
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest with class balancing."""
        
        # Skip SMOTE for now due to compatibility issues, use class weights instead
        logger.info("Using class weights for imbalanced data handling")
        X_resampled, y_resampled = X, y
        
        # Compute class weights
        classes = np.unique(y_resampled)
        if len(y_resampled.shape) > 1:
            y_resampled = y_resampled.ravel()
        class_weights = compute_class_weight('balanced', classes=classes, y=y_resampled)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Train Random Forest with optimized parameters
        model = RandomForestClassifier(
            n_estimators=1000,      # More trees for better performance
            max_depth=15,           # Reduce depth to prevent overfitting
            min_samples_split=10,   # Increase to reduce overfitting
            min_samples_leaf=5,     # Increase to reduce overfitting
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            max_features='log2',    # Use log2 for better feature selection
            criterion='gini',       # Gini for better fraud detection
            warm_start=False
        )
        
        model.fit(X_resampled, y_resampled)
        
        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance['random_forest'] = dict(zip(
                self.feature_columns, model.feature_importances_
            ))
        
        logger.info("Random Forest training completed")
        return model
    
    def predict_anomalies(self, X: pd.DataFrame, method: str = 'ensemble') -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using specified method."""
        
        # Check if scaler was fitted with different number of features
        try:
            X_scaled = self.scaler.transform(X)
        except ValueError as e:
            if "features" in str(e):
                logger.warning(f"Feature dimension mismatch detected. Retraining models with current feature set.")
                # Retrain with current features
                y = None  # No labels available for retraining
                self.train_models(X, y)
                X_scaled = self.scaler.transform(X)
            else:
                raise e
        
        if method == 'isolation_forest':
            return self._predict_isolation_forest(X_scaled)
        elif method == 'random_forest':
            return self._predict_random_forest(X_scaled)
        elif method == 'ensemble':
            return self._predict_ensemble(X_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _predict_isolation_forest(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using Isolation Forest."""
        
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest not trained")
        
        anomaly_scores = self.isolation_forest.decision_function(X)
        predictions = self.isolation_forest.predict(X)
        predictions = (predictions == -1).astype(int)
        
        return predictions, -anomaly_scores  # Invert scores for consistency
    
    def _predict_random_forest(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using Random Forest."""
        
        if self.random_forest is None:
            raise ValueError("Random Forest not trained")
        
        predictions = self.random_forest.predict(X)
        probabilities = self.random_forest.predict_proba(X)[:, 1]  # Probability of fraud
        
        return predictions, probabilities
    
    def _predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using ensemble of models."""
        
        # Get predictions from both models
        iso_pred, iso_scores = self._predict_isolation_forest(X)
        
        if self.random_forest is not None:
            rf_pred, rf_scores = self._predict_random_forest(X)
            
            # Normalize scores to [0, 1]
            iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)
            rf_scores_norm = rf_scores
            
            # Weighted ensemble
            ensemble_scores = (
                self.ensemble_weights['isolation_forest'] * iso_scores_norm +
                self.ensemble_weights['random_forest'] * rf_scores_norm
            )
            
            # Ensemble predictions using optimized threshold for better precision
            threshold = np.percentile(ensemble_scores, 97)  # Use 97th percentile for better precision
            ensemble_pred = (ensemble_scores > threshold).astype(int)
            
        else:
            # Use only Isolation Forest if Random Forest not available
            ensemble_pred, ensemble_scores = iso_pred, iso_scores
        
        return ensemble_pred, ensemble_scores
    
    def evaluate_model(self, X: pd.DataFrame, y: np.ndarray, method: str = 'ensemble') -> Dict:
        """Evaluate model performance."""
        
        # Use the same feature preparation as training
        X_prepared = self.prepare_features(X, self.feature_columns)
        predictions, scores = self.predict_anomalies(X, method)
        
        # Ensure consistent lengths
        min_len = min(len(y), len(predictions))
        y_eval = y[:min_len]
        predictions_eval = predictions[:min_len]
        scores_eval = scores[:min_len]
        
        # Calculate metrics
        precision = precision_score(y_eval, predictions_eval, zero_division=0)
        recall = recall_score(y_eval, predictions_eval, zero_division=0)
        f1 = f1_score(y_eval, predictions_eval, zero_division=0)
        
        try:
            auc = roc_auc_score(y_eval, scores_eval)
        except ValueError:
            auc = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def save_models(self, model_dir: str = "models/"):
        """Save trained models."""
        
        if self.isolation_forest is not None:
            joblib.dump(self.isolation_forest, f"{model_dir}enhanced_isolation_forest.pkl")
        
        if self.random_forest is not None:
            joblib.dump(self.random_forest, f"{model_dir}enhanced_random_forest.pkl")
        
        joblib.dump(self.scaler, f"{model_dir}enhanced_scaler.pkl")
        joblib.dump(self.feature_columns, f"{model_dir}enhanced_feature_columns.pkl")
        
        if self.feature_importance:
            joblib.dump(self.feature_importance, f"{model_dir}enhanced_feature_importance.pkl")
        
        logger.info("Enhanced models saved successfully")
    
    def load_models(self, model_dir: str = "models/") -> bool:
        """Load trained models."""
        
        try:
            import os
            
            if os.path.exists(f"{model_dir}enhanced_isolation_forest.pkl"):
                self.isolation_forest = joblib.load(f"{model_dir}enhanced_isolation_forest.pkl")
                self.models['isolation_forest'] = self.isolation_forest
            
            if os.path.exists(f"{model_dir}enhanced_random_forest.pkl"):
                self.random_forest = joblib.load(f"{model_dir}enhanced_random_forest.pkl")
                self.models['random_forest'] = self.random_forest
            
            if os.path.exists(f"{model_dir}enhanced_scaler.pkl"):
                self.scaler = joblib.load(f"{model_dir}enhanced_scaler.pkl")
            
            if os.path.exists(f"{model_dir}enhanced_feature_columns.pkl"):
                self.feature_columns = joblib.load(f"{model_dir}enhanced_feature_columns.pkl")
            
            if os.path.exists(f"{model_dir}enhanced_feature_importance.pkl"):
                self.feature_importance = joblib.load(f"{model_dir}enhanced_feature_importance.pkl")
            
            logger.info("Enhanced models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading enhanced models: {e}")
            return False
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained models."""
        return self.feature_importance
