"""SHAP-based explainability for anomaly detection models."""

import pandas as pd
import numpy as np
import shap
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Provides SHAP-based explanations for anomaly detection models."""
    
    def __init__(self):
        self.explainers = {}
        self.surrogate_models = {}
        self.feature_names = []
        
    def create_surrogate_model(self, X: np.ndarray, anomaly_scores: np.ndarray, 
                             feature_names: List[str]) -> GradientBoostingClassifier:
        """Create a surrogate model to approximate the anomaly detector."""
        
        logger.info("Creating surrogate model for explainability...")
        
        # Convert anomaly scores to binary labels using threshold
        threshold = np.percentile(anomaly_scores, 90)  # Top 10% as anomalies
        y_binary = (anomaly_scores > threshold).astype(int)
        
        # Train gradient boosting surrogate
        surrogate = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Split data for training surrogate
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        surrogate.fit(X_train, y_train)
        
        # Evaluate surrogate performance
        train_score = surrogate.score(X_train, y_train)
        test_score = surrogate.score(X_test, y_test)
        
        logger.info(f"Surrogate model performance - Train: {train_score:.3f}, Test: {test_score:.3f}")
        
        self.feature_names = feature_names
        return surrogate
    
    def create_shap_explainer(self, surrogate_model: GradientBoostingClassifier, 
                            X_background: np.ndarray) -> shap.TreeExplainer:
        """Create SHAP explainer for the surrogate model."""
        
        logger.info("Creating SHAP explainer...")
        
        # Use a sample of background data for efficiency
        background_sample = shap.sample(X_background, min(100, len(X_background)))
        
        # Create TreeExplainer for gradient boosting
        explainer = shap.TreeExplainer(surrogate_model, background_sample)
        
        return explainer
    
    def explain_prediction(self, explainer: shap.TreeExplainer, X_instance: np.ndarray, 
                         top_k: int = 10) -> Dict[str, Any]:
        """Explain a single prediction."""
        
        # Get SHAP values
        shap_values = explainer.shap_values(X_instance.reshape(1, -1))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, take positive class
        
        shap_values = shap_values[0]  # Get values for single instance
        
        # Create explanation dictionary
        feature_importance = list(zip(self.feature_names, shap_values, X_instance))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        explanation = {
            'top_features': feature_importance[:top_k],
            'shap_values': shap_values,
            'feature_values': X_instance,
            'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0
        }
        
        return explanation
    
    def explain_batch(self, explainer: shap.TreeExplainer, X_batch: np.ndarray) -> np.ndarray:
        """Explain a batch of predictions."""
        
        shap_values = explainer.shap_values(X_batch)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        return shap_values
    
    def create_feature_importance_summary(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Create global feature importance summary."""
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create summary dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def generate_alert_explanation(self, transaction_data: Dict, explanation: Dict[str, Any]) -> str:
        """Generate human-readable explanation for an alert."""
        
        top_features = explanation['top_features']
        
        explanation_text = f"**Fraud Alert for Transaction {transaction_data.get('transaction_id', 'Unknown')}**\n\n"
        explanation_text += f"**Account:** {transaction_data.get('account_id', 'Unknown')}\n"
        explanation_text += f"**Amount:** ${transaction_data.get('amount', 0):.2f}\n"
        explanation_text += f"**Merchant:** {transaction_data.get('merchant_name', 'Unknown')}\n"
        explanation_text += f"**Time:** {transaction_data.get('timestamp', 'Unknown')}\n\n"
        
        explanation_text += "**Key Risk Factors:**\n"
        
        for i, (feature, shap_val, feature_val) in enumerate(top_features[:5]):
            impact = "increases" if shap_val > 0 else "decreases"
            explanation_text += f"{i+1}. **{self._format_feature_name(feature)}**: {feature_val:.2f} ({impact} risk by {abs(shap_val):.3f})\n"
        
        return explanation_text
    
    def _format_feature_name(self, feature: str) -> str:
        """Format feature names for human readability."""
        
        formatting_map = {
            'amount_abs': 'Transaction Amount',
            'hour_of_day': 'Hour of Day',
            'is_night': 'Night Transaction',
            'is_weekend': 'Weekend Transaction',
            'txn_count_1d': '24h Transaction Count',
            'txn_count_7d': '7-day Transaction Count',
            'velocity_7d_vs_30d': '7d vs 30d Spending Velocity',
            'is_new_merchant': 'New Merchant',
            'time_since_last_txn': 'Time Since Last Transaction',
            'amount_zscore_7d': 'Amount Z-score (7d)',
            'unique_merchants_7d': 'Merchant Diversity (7d)',
            'rapid_succession': 'Rapid Successive Transactions',
            'is_different_city': 'Different City',
            'category_entropy_7d': 'Category Diversity (7d)'
        }
        
        return formatting_map.get(feature, feature.replace('_', ' ').title())

def main():
    """Main function for creating model explanations."""
    
    logging.basicConfig(level=logging.INFO)
    
    from sqlalchemy import create_engine
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.anomaly_detection import AnomalyDetector
    
    # Load data
    engine = create_engine("sqlite:///data/transactions.db")
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM feature_store", conn)
        
        if len(df) == 0:
            print("No feature data found. Run feature engineering first.")
            return
        
        # Initialize components
        detector = AnomalyDetector()
        explainer = ModelExplainer()
        
        # Load trained models
        if not detector.load_models():
            print("Error: No trained models found")
            return
        
        # Get feature columns (exclude non-numeric and ID columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['transaction_id', 'account_id', 'timestamp', 'is_fraud', 'unix_timestamp']
        feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        X = detector.prepare_features(df, feature_columns)
        detector.train_isolation_forest(X)
        detector.save_models()
        
        # Get predictions and scores
        feature_columns = detector.get_feature_columns()
        X = detector.prepare_features(df, feature_columns)
        predictions, scores = detector.predict_anomalies(X, 'isolation_forest')
        
        # Create surrogate model for explainability
        surrogate = explainer.create_surrogate_model(X, scores, detector.feature_columns)
        explainer.surrogate_models['isolation_forest'] = surrogate
        
        # Create SHAP explainer
        shap_explainer = explainer.create_shap_explainer(surrogate, X)
        explainer.explainers['isolation_forest'] = shap_explainer
        
        # Generate explanations for anomalous transactions
        anomaly_indices = np.where(predictions == 1)[0]
        
        if len(anomaly_indices) > 0:
            print(f"\nGenerating explanations for {len(anomaly_indices)} anomalous transactions...")
            
            # Explain top 5 anomalies
            for i, idx in enumerate(anomaly_indices[:5]):
                explanation = explainer.explain_prediction(shap_explainer, X[idx])
                
                # Get transaction data
                transaction_data = df.iloc[idx].to_dict()
                
                # Generate readable explanation
                alert_text = explainer.generate_alert_explanation(transaction_data, explanation)
                print(f"\n--- Anomaly {i+1} ---")
                print(alert_text)
        
        # Global feature importance
        batch_shap = explainer.explain_batch(shap_explainer, X[:1000])  # Sample for efficiency
        importance_df = explainer.create_feature_importance_summary(batch_shap)
        
        print(f"\n**Global Feature Importance:**")
        print(importance_df.head(10))
        
        # Save explainer
        joblib.dump(explainer, "models/explainer.pkl")
        print("\nExplainer saved successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run the full pipeline first.")

if __name__ == "__main__":
    main()
