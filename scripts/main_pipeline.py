"""
Enhanced Pipeline with Advanced Feature Engineering and Ensemble Models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
from datetime import datetime

# Import enhanced components
from src.features.advanced_features import AdvancedFeatureEngineer
from src.models.balance_forecasting import SimpleForecaster
from src.models.enhanced_anomaly_detection import EnhancedAnomalyDetector
from src.ingestion.data_loader import TransactionDataLoader
from src.evaluation.model_evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_enhanced_pipeline():
    """Run the enhanced fraud detection pipeline."""
    
    print("Starting Enhanced Customer Transaction Intelligence Pipeline")
    print("=" * 70)
    
    try:
        # Initialize components
        data_loader = TransactionDataLoader()
        feature_engineer = AdvancedFeatureEngineer()
        detector = EnhancedAnomalyDetector(contamination=0.02)
        evaluator = ModelEvaluator()
        
        # Load data
        print("Loading transaction data...")
        engine = create_engine("sqlite:///data/transactions.db")
        
        with engine.connect() as conn:
            # Load raw transactions with fraud labels
            df = pd.read_sql("""
                SELECT t.*, COALESCE(t.is_fraud, 0) as is_fraud
                FROM transactions_raw t 
                ORDER BY t.timestamp
                LIMIT 50000
            """, conn)
            
            # Reset index to ensure clean indexing
            df = df.reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} transactions for enhanced processing")
        
        # Advanced feature engineering
        print("Creating advanced features...")
        df_features = feature_engineer.create_features(df)
        
        # Get enhanced feature columns
        feature_columns = feature_engineer.get_feature_columns()
        available_features = [col for col in feature_columns if col in df_features.columns]
        
        logger.info(f"Created {len(available_features)} advanced features")
        
        # Prepare features and labels - ensure consistent indexing
        X = detector.prepare_features(df_features, available_features)
        if 'is_fraud' in df_features.columns:
            # Align y with X after feature selection
            y = df_features.iloc[X.index]['is_fraud'].values
            # Ensure y is 1D
            if len(y.shape) > 1:
                y = y.ravel()
            print(f"Data alignment: X shape={X.shape}, y shape={y.shape}")
        else:
            y = None
        
        # Train enhanced models
        print("Training enhanced ensemble models...")
        models = detector.train_models(X, y)
        
        # Save enhanced models
        detector.save_models()
        
        # Evaluate models
        if y is not None:
            print("Evaluating model performance...")
            
            # Evaluate different methods
            methods = ['isolation_forest', 'ensemble']
            if detector.random_forest is not None:
                methods.append('random_forest')
            
            results = {}
            for method in methods:
                metrics = detector.evaluate_model(X, y, method)
                results[method] = metrics
                
                print(f"\n{method.upper()} Performance:")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall: {metrics['recall']:.3f}")
                print(f"  F1 Score: {metrics['f1']:.3f}")
                print(f"  AUC: {metrics['auc']:.3f}")
            
            # Save best results
            best_method = max(results.keys(), key=lambda k: results[k]['f1'])
            print(f"\nBest performing method: {best_method}")
            
            # Store evaluation results
            with engine.connect() as conn:
                # Create enhanced evaluation table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS enhanced_model_evaluation (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        method TEXT,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        auc REAL
                    )
                """))
                
                # Insert results
                for method, metrics in results.items():
                    conn.execute(text("""
                        INSERT INTO enhanced_model_evaluation 
                        (method, precision, recall, f1_score, auc)
                        VALUES (:method, :precision, :recall, :f1_score, :auc)
                    """), {
                        'method': method, 
                        'precision': metrics['precision'], 
                        'recall': metrics['recall'], 
                        'f1_score': metrics['f1'], 
                        'auc': metrics['auc']
                    })
                
                conn.commit()
        
        # Generate predictions for dashboard
        print("Generating predictions for dashboard...")
        predictions, scores = detector.predict_anomalies(X, method='ensemble')
        
        # Store enhanced predictions
        df_results = df_features.copy()
        df_results['is_anomaly_enhanced'] = predictions
        df_results['anomaly_score_enhanced'] = scores
        
        # Save to database
        with engine.connect() as conn:
            # Create enhanced results table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS enhanced_anomaly_results (
                    transaction_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    account_id TEXT,
                    amount REAL,
                    merchant_name TEXT,
                    is_anomaly_enhanced INTEGER,
                    anomaly_score_enhanced REAL,
                    risk_score REAL
                )
            """))
            
            # Insert results
            result_cols = ['transaction_id', 'timestamp', 'account_id', 'amount', 
                          'merchant_name', 'is_anomaly_enhanced', 'anomaly_score_enhanced']
            
            if 'risk_score' in df_results.columns:
                result_cols.append('risk_score')
            else:
                df_results['risk_score'] = 0.5
                result_cols.append('risk_score')
            
            available_result_cols = [col for col in result_cols if col in df_results.columns]
            
            df_results[available_result_cols].to_sql(
                'enhanced_anomaly_results', 
                conn, 
                if_exists='replace', 
                index=False
            )
        
        # Feature importance analysis
        if detector.feature_importance:
            print("\nTop 10 Most Important Features:")
            for model_name, importance_dict in detector.feature_importance.items():
                print(f"\n{model_name.upper()}:")
                sorted_features = sorted(importance_dict.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
                for feature, importance in sorted_features:
                    print(f"  {feature}: {importance:.4f}")
        
        print("\n" + "=" * 70)
        print("Enhanced Pipeline completed successfully!")
        print("\nEnhancements implemented:")
        print("* Advanced feature engineering (velocity, risk scoring, merchant analysis)")
        print("* Ensemble methods (Isolation Forest + Random Forest)")
        print("* Class balancing with class weights")
        print("* Feature selection and importance analysis")
        print("* Comprehensive evaluation framework")
        
        anomaly_count = predictions.sum()
        anomaly_rate = predictions.mean()
        print(f"\nResults: {anomaly_count:,} anomalies detected ({anomaly_rate:.2%} rate)")
        
        if y is not None and len(results) > 0:
            best_metrics = results[best_method]
            print(f"Best model ({best_method}):")
            print(f"  Precision: {best_metrics['precision']:.3f}")
            print(f"  Recall: {best_metrics['recall']:.3f}")
            print(f"  F1 Score: {best_metrics['f1']:.3f}")
            print(f"  AUC: {best_metrics['auc']:.3f}")
        
        print("\nNext Steps:")
        print("1. Start the dashboard: streamlit run dashboard/app.py")
        print("2. View enhanced model performance")
        print("3. Analyze feature importance and risk factors")
        
    except Exception as e:
        logger.error(f"Enhanced pipeline failed: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    run_enhanced_pipeline()
