"""Model evaluation framework for the transaction intelligence system."""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for fraud detection and forecasting."""
    
    def __init__(self):
        self.engine = create_engine("sqlite:///data/transactions.db")
        self.results = {}
    
    def evaluate_anomaly_detection(self) -> dict:
        """Evaluate anomaly detection model performance."""
        
        try:
            # Load feature data
            with self.engine.connect() as conn:
                df = pd.read_sql("SELECT * FROM feature_store", conn)
            
            if 'is_fraud' not in df.columns:
                logger.warning("No fraud labels available for evaluation")
                return {}
            
            # Load trained model
            from models.fraud_detection import AnomalyDetector
            detector = AnomalyDetector()
            
            # Get feature columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['is_fraud', 'unix_timestamp']
            feature_columns = [col for col in numeric_cols if col not in exclude_cols]
            
            # Prepare features
            X = detector.prepare_features(df, feature_columns)
            
            # Load or train model
            models_loaded = detector.load_models()
            if not models_loaded or detector.isolation_forest is None:
                # Train improved model with optimized parameters
                from sklearn.ensemble import IsolationForest
                detector.isolation_forest = IsolationForest(
                    contamination=0.03,
                    n_estimators=1000,
                    max_samples=0.5,
                    max_features=0.8,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )
                detector.isolation_forest.fit(X)
                detector.save_models()
            
            # Get predictions
            predictions, scores = detector.predict_anomalies(X, 'isolation_forest')
            y_true = df['is_fraud'].values
            
            # Calculate metrics
            metrics = {
                'precision': precision_score(y_true, predictions, zero_division=0),
                'recall': recall_score(y_true, predictions, zero_division=0),
                'f1': f1_score(y_true, predictions, zero_division=0),
                'auc': roc_auc_score(y_true, scores) if len(np.unique(scores)) > 1 else 0,
                'alert_rate': predictions.mean(),
                'fraud_rate': y_true.mean(),
                'true_positives': ((y_true == 1) & (predictions == 1)).sum(),
                'false_positives': ((y_true == 0) & (predictions == 1)).sum(),
                'true_negatives': ((y_true == 0) & (predictions == 0)).sum(),
                'false_negatives': ((y_true == 1) & (predictions == 0)).sum()
            }
            
            self.results['anomaly_detection'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating anomaly detection: {e}")
            return {}
    
    def evaluate_forecasting(self) -> dict:
        """Evaluate forecasting model performance."""
        
        try:
            # Load forecast data
            with self.engine.connect() as conn:
                forecasts = pd.read_sql("SELECT * FROM balance_forecasts", conn)
            
            if len(forecasts) == 0:
                logger.warning("No forecast data available")
                return {}
            
            # Calculate forecast accuracy metrics (simplified)
            metrics = {
                'total_forecasts': len(forecasts),
                'unique_accounts': forecasts['account_id'].nunique(),
                'avg_forecast_value': forecasts['forecast'].mean(),
                'overdraft_risk_accounts': forecasts[forecasts['overdraft_risk'] == 1]['account_id'].nunique(),
                'forecast_horizon_days': 30
            }
            
            # If we had actual future data, we would calculate MAPE, MAE, RMSE here
            # For now, we'll use the forecast spread as a proxy for uncertainty
            metrics['avg_forecast_spread'] = (forecasts['upper_bound'] - forecasts['lower_bound']).mean()
            
            self.results['forecasting'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating forecasting: {e}")
            return {}
    
    def evaluate_system_performance(self) -> dict:
        """Evaluate overall system performance."""
        
        try:
            # Load transaction data
            with self.engine.connect() as conn:
                df = pd.read_sql("SELECT COUNT(*) as total_transactions FROM feature_store", conn)
                total_transactions = df['total_transactions'].iloc[0]
            
            # System metrics
            metrics = {
                'total_transactions_processed': total_transactions,
                'pipeline_completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'models_trained': len([f for f in os.listdir('models/') if f.endswith('.pkl')]),
                'database_tables': self._count_database_tables()
            }
            
            self.results['system'] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating system: {e}")
            return {}
    
    def _count_database_tables(self) -> int:
        """Count number of tables in database."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                return len(result.fetchall())
        except:
            return 0
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report."""
        
        # Run all evaluations
        anomaly_metrics = self.evaluate_anomaly_detection()
        forecast_metrics = self.evaluate_forecasting()
        system_metrics = self.evaluate_system_performance()
        
        # Generate report
        report = "# Customer Transaction Intelligence - Evaluation Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Anomaly Detection Results
        report += "## Anomaly Detection Performance\n\n"
        if anomaly_metrics:
            report += f"- **Precision:** {anomaly_metrics['precision']:.3f}\n"
            report += f"- **Recall:** {anomaly_metrics['recall']:.3f}\n"
            report += f"- **F1 Score:** {anomaly_metrics['f1']:.3f}\n"
            report += f"- **AUC:** {anomaly_metrics['auc']:.3f}\n"
            report += f"- **Alert Rate:** {anomaly_metrics['alert_rate']:.2%}\n"
            report += f"- **Fraud Rate:** {anomaly_metrics['fraud_rate']:.2%}\n"
            
            # Performance assessment
            if anomaly_metrics['precision'] >= 0.85 and anomaly_metrics['recall'] >= 0.70:
                report += "\n**Status:** Excellent performance - meets production criteria\n"
            elif anomaly_metrics['f1'] >= 0.60:
                report += "\n**Status:** Good performance - minor tuning recommended\n"
            else:
                report += "\n**Status:** Needs improvement - significant tuning required\n"
        else:
            report += "Evaluation failed\n"
        
        # Forecasting Results
        report += "\n## Balance Forecasting Performance\n\n"
        if forecast_metrics:
            report += f"- **Total Forecasts:** {forecast_metrics['total_forecasts']:,}\n"
            report += f"- **Accounts Covered:** {forecast_metrics['unique_accounts']}\n"
            report += f"- **Overdraft Risk Accounts:** {forecast_metrics['overdraft_risk_accounts']}\n"
            report += f"- **Forecast Horizon:** {forecast_metrics['forecast_horizon_days']} days\n"
            report += f"- **Average Forecast:** ${forecast_metrics['avg_forecast_value']:.2f}\n"
            
            report += "\n**Status:** Forecasting pipeline operational\n"
        else:
            report += "Evaluation failed\n"
        
        # System Performance
        report += "\n## System Performance\n\n"
        if system_metrics:
            report += f"- **Transactions Processed:** {system_metrics['total_transactions_processed']:,}\n"
            report += f"- **Models Trained:** {system_metrics['models_trained']}\n"
            report += f"- **Database Tables:** {system_metrics['database_tables']}\n"
            report += f"- **Pipeline Completed:** {system_metrics['pipeline_completion_time']}\n"
            
            report += "\n**Status:** System fully operational\n"
        
        # Recommendations
        report += "\n## Recommendations\n\n"
        
        if anomaly_metrics and anomaly_metrics['precision'] < 0.85:
            report += "- **Improve Precision:** Consider feature selection, threshold tuning, or ensemble methods\n"
        
        if anomaly_metrics and anomaly_metrics['recall'] < 0.70:
            report += "- **Improve Recall:** Lower detection threshold or add more behavioral features\n"
        
        report += "- **Production Deployment:** Set up real-time scoring API and alert system\n"
        report += "- **Monitoring:** Implement model drift detection and automated retraining\n"
        report += "- **Explainability:** Deploy SHAP explanations for all alerts\n"
        
        return report

def main():
    """Main evaluation function."""
    
    logging.basicConfig(level=logging.INFO)
    
    evaluator = ModelEvaluator()
    report = evaluator.generate_evaluation_report()
    
    print(report)
    
    # Save report
    with open('evaluation_report.md', 'w') as f:
        f.write(report)
    
    print("\nEvaluation report saved to evaluation_report.md")

if __name__ == "__main__":
    main()
