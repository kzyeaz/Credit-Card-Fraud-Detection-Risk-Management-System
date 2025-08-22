"""Simplified feature engineering for large datasets."""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

class SimpleFeatureEngineer:
    """Simplified feature engineering for performance."""
    
    def __init__(self, db_connection_string: str = "sqlite:///data/transactions.db"):
        self.engine = create_engine(db_connection_string)
    
    def create_features(self, sample_size: int = 10000) -> pd.DataFrame:
        """Create features from a sample of the data."""
        
        # Load sample data
        query = f"""
        SELECT * FROM transactions_raw 
        ORDER BY RANDOM() 
        LIMIT {sample_size}
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        
        # Amount features
        df['amount_abs'] = df['amount'].abs()
        df['amount_log'] = np.log1p(df['amount_abs'])
        df['is_large_amount'] = (df['amount_abs'] > df['amount_abs'].quantile(0.95)).astype(int)
        
        # Simple aggregated features per account
        account_stats = df.groupby('account_id').agg({
            'amount': ['mean', 'std', 'count'],
            'merchant_name': 'nunique',
            'merchant_category': 'nunique'
        }).reset_index()
        
        # Flatten column names
        account_stats.columns = [
            'account_id', 'avg_amount', 'std_amount', 'txn_count',
            'unique_merchants', 'unique_categories'
        ]
        
        # Merge back
        df = df.merge(account_stats, on='account_id', how='left')
        
        # Z-scores
        df['amount_zscore'] = (df['amount_abs'] - df['avg_amount']) / (df['std_amount'] + 1e-6)
        
        # Merchant frequency
        merchant_freq = df['merchant_name'].value_counts()
        df['merchant_frequency'] = df['merchant_name'].map(merchant_freq)
        df['is_rare_merchant'] = (df['merchant_frequency'] <= 5).astype(int)
        
        # Category encoding
        category_mapping = {cat: i for i, cat in enumerate(df['merchant_category'].unique())}
        df['category_encoded'] = df['merchant_category'].map(category_mapping)
        
        # Select final feature columns
        feature_cols = [
            'transaction_id', 'account_id', 'timestamp', 'amount', 'is_fraud',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_night',
            'amount_abs', 'amount_log', 'is_large_amount',
            'avg_amount', 'std_amount', 'txn_count', 'unique_merchants', 'unique_categories',
            'amount_zscore', 'merchant_frequency', 'is_rare_merchant', 'category_encoded'
        ]
        
        # Add V features if they exist
        v_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
        feature_cols.extend(v_cols)
        
        # Return only available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        result_df = df[available_cols].copy()
        
        # Fill any remaining NaN values
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        result_df[numeric_cols] = result_df[numeric_cols].fillna(0)
        
        return result_df

def main():
    """Main function for simplified feature engineering."""
    
    logging.basicConfig(level=logging.INFO)
    
    fe = SimpleFeatureEngineer()
    features_df = fe.create_features(sample_size=20000)
    
    if len(features_df) > 0:
        # Save to database
        with fe.engine.connect() as conn:
            features_df.to_sql('feature_store', conn, if_exists='replace', index=False)
        
        print(f"Feature engineering completed!")
        print(f"Generated {len(features_df)} feature records")
        print(f"Feature columns: {len(features_df.columns)}")
        print(f"Fraud rate: {features_df['is_fraud'].mean():.2%}")
        
    else:
        print("No features generated")

if __name__ == "__main__":
    main()
