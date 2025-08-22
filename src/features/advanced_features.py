"""
Advanced Feature Engineering for Fraud Detection
Implements sophisticated features to improve model performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for fraud detection."""
    
    def __init__(self):
        self.merchant_stats = {}
        self.account_profiles = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for fraud detection."""
        
        logger.info("Creating advanced features for fraud detection...")
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['account_id', 'timestamp'])
        
        # Time-based features
        df = self._create_time_features(df)
        
        # Transaction velocity features
        df = self._create_velocity_features(df)
        
        # Amount-based features
        df = self._create_amount_features(df)
        
        # Merchant and location features
        df = self._create_merchant_features(df)
        
        # Account behavior features
        df = self._create_account_features(df)
        
        # Risk scoring features
        df = self._create_risk_features(df)
        
        logger.info(f"Created {len([c for c in df.columns if c.startswith(('txn_', 'amt_', 'merch_', 'risk_', 'vel_'))])} advanced features")
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        return df
    
    def _create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction velocity features."""
        
        # Sort by account and timestamp
        df = df.sort_values(['account_id', 'timestamp']).reset_index(drop=True)
        
        # Time since last transaction
        df['time_since_last'] = df.groupby('account_id')['timestamp'].diff().dt.total_seconds() / 3600
        df['time_since_last'] = df['time_since_last'].fillna(24)  # Default 24 hours
        
        # Simple velocity features using shift to avoid rolling window issues
        df['prev_amount'] = df.groupby('account_id')['amount'].shift(1).fillna(0)
        df['amount_change'] = df['amount'] - df['prev_amount']
        df['amount_ratio'] = df['amount'] / (df['prev_amount'] + 1)
        
        # Count transactions in last N transactions
        for n in [5, 10, 20]:
            df[f'txn_count_last_{n}'] = df.groupby('account_id').cumcount()
            df[f'txn_count_last_{n}'] = df[f'txn_count_last_{n}'].apply(lambda x: min(x, n))
        
        # Average amounts in last N transactions
        for n in [5, 10]:
            df[f'avg_amount_last_{n}'] = df.groupby('account_id')['amount'].transform(
                lambda x: x.rolling(window=n, min_periods=1).mean().shift(1).fillna(x.mean())
            )
        
        # Velocity ratios
        df['vel_recent_vs_avg'] = df['amount'] / (df['avg_amount_last_10'] + 1)
        
        # Rapid fire transactions (< 5 minutes apart)
        df['rapid_fire'] = (df['time_since_last'] < 0.083).astype(int)  # 5 minutes = 0.083 hours
        
        return df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features."""
        
        # Simple amount statistics using rolling windows without time-based indexing
        for window in [5, 10, 20]:
            df[f'amount_mean_last_{window}'] = df.groupby('account_id')['amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1).fillna(x.mean())
            )
            df[f'amount_std_last_{window}'] = df.groupby('account_id')['amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().shift(1).fillna(0)
            )
            df[f'amount_max_last_{window}'] = df.groupby('account_id')['amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max().shift(1).fillna(x.mean())
            )
        
        # Amount anomaly scores
        df['amount_zscore'] = df.groupby('account_id')['amount'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        # Amount percentiles within account
        df['amount_percentile'] = df.groupby('account_id')['amount'].transform(
            lambda x: x.rank(pct=True)
        )
        
        # Amount deviation from recent average
        df['amount_vs_recent_avg'] = df['amount'] / (df['amount_mean_last_10'] + 1)
        df['amount_vs_recent_max'] = df['amount'] / (df['amount_max_last_10'] + 1)
        
        # Round number detection
        df['is_round_amount'] = (df['amount'] % 10 == 0).astype(int)
        df['is_very_round'] = (df['amount'] % 100 == 0).astype(int)
        
        return df
    
    def _create_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create merchant and location features."""
        
        # Check if merchant columns exist
        if 'merchant_name' in df.columns:
            # Merchant frequency for account
            merchant_freq = df.groupby(['account_id', 'merchant_name']).size().reset_index(name='freq')
            df = df.merge(merchant_freq, on=['account_id', 'merchant_name'], how='left')
            df['merch_freq_account'] = df['freq']
            df = df.drop('freq', axis=1)
            
            # Merchant statistics - simplified approach
            merchant_stats = df.groupby('merchant_name')['amount'].agg(['mean', 'std', 'count']).reset_index()
            merchant_stats.columns = ['merchant_name', 'merch_amt_mean', 'merch_amt_std', 'merch_txn_count']
            merchant_stats = merchant_stats.fillna(0)
            
            df = df.merge(merchant_stats, on='merchant_name', how='left')
            
            # Merchant fraud rate if fraud labels available
            if 'is_fraud' in df.columns:
                fraud_stats = df.groupby('merchant_name')['is_fraud'].mean().reset_index()
                df = df.merge(fraud_stats, on='merchant_name', how='left', suffixes=('', '_fraud'))
                df = df.rename(columns={'is_fraud_fraud': 'merch_fraud_rate'})
            else:
                df['merch_fraud_rate'] = 0.0
        else:
            # Create default values when merchant_name is not available
            df['merch_freq_account'] = 1
            df['merch_amt_mean'] = df['amount'].mean()
            df['merch_amt_std'] = df['amount'].std()
            df['merch_txn_count'] = 1
            df['merch_fraud_rate'] = 0.0
        
        # Merchant category encoding
        if 'merchant_category' in df.columns:
            category_freq = df['merchant_category'].value_counts()
            df['merch_cat_freq'] = df['merchant_category'].map(category_freq)
            
            # Category risk score
            if 'is_fraud' in df.columns:
                category_risk = df.groupby('merchant_category')['is_fraud'].mean().to_dict()
                df['merch_cat_risk'] = df['merchant_category'].map(category_risk)
            else:
                df['merch_cat_risk'] = 0.5  # Neutral risk
        else:
            # Create default values when merchant_category is not available
            df['merch_cat_freq'] = 1
            df['merch_cat_risk'] = 0.5
        
        # New merchant indicator
        df['is_new_merchant'] = (df['merch_freq_account'] == 1).astype(int)
        
        return df
    
    def _create_account_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create account behavior features."""
        
        # Account age (days since first transaction)
        first_txn = df.groupby('account_id')['timestamp'].min()
        df['account_age_days'] = (df['timestamp'] - df['account_id'].map(first_txn)).dt.days
        
        # Spending patterns
        daily_spending = df.groupby(['account_id', df['timestamp'].dt.date])['amount'].sum()
        avg_daily_spending = daily_spending.groupby('account_id').mean()
        df['avg_daily_spending'] = df['account_id'].map(avg_daily_spending)
        
        # Transaction diversity
        if 'merchant_name' in df.columns:
            unique_merchants = df.groupby('account_id')['merchant_name'].nunique()
            df['merchant_diversity'] = df['account_id'].map(unique_merchants)
        else:
            df['merchant_diversity'] = 1
        
        if 'merchant_category' in df.columns:
            unique_categories = df.groupby('account_id')['merchant_category'].nunique()
            df['category_diversity'] = df['account_id'].map(unique_categories)
        else:
            df['category_diversity'] = 1
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk features."""
        
        # High-risk time indicator
        df['risk_time'] = ((df['is_night'] == 1) | (df['is_weekend'] == 1)).astype(int)
        
        # Amount anomaly risk
        if 'amount_zscore' in df.columns:
            df['risk_amount_anomaly'] = (df['amount_zscore'].abs() > 2).astype(int)
        else:
            df['risk_amount_anomaly'] = 0
        
        # Velocity risk
        if 'rapid_fire' in df.columns and 'time_since_last' in df.columns:
            df['risk_velocity'] = (
                (df['rapid_fire'] == 1) |
                (df['time_since_last'] < 0.017)  # Less than 1 minute
            ).astype(int)
        else:
            df['risk_velocity'] = 0
        
        # New merchant risk
        if 'is_new_merchant' in df.columns and 'amount_mean_last_10' in df.columns:
            df['risk_new_merchant'] = (
                (df['is_new_merchant'] == 1) & 
                (df['amount'] > df['amount_mean_last_10'])
            ).astype(int)
        else:
            df['risk_new_merchant'] = 0
        
        # Composite risk score using available features
        risk_components = []
        
        if 'amount_zscore' in df.columns:
            risk_components.append(df['amount_zscore'].abs() > 2)
        if 'rapid_fire' in df.columns:
            risk_components.append(df['rapid_fire'] == 1)
        if 'is_night' in df.columns:
            risk_components.append(df['is_night'] == 1)
        if 'is_new_merchant' in df.columns:
            risk_components.append(df['is_new_merchant'] == 1)
        if 'vel_recent_vs_avg' in df.columns:
            risk_components.append(df['vel_recent_vs_avg'] > 3)
        
        if risk_components:
            df['risk_score'] = sum(risk_components) / len(risk_components)
        else:
            df['risk_score'] = 0.5
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of engineered feature columns."""
        
        base_features = ['amount', 'hour', 'day_of_week', 'month']
        
        advanced_features = [
            # Time features
            'is_weekend', 'is_night', 'is_business_hours',
            
            # Velocity features
            'time_since_last', 'txn_count_1H', 'txn_count_6H', 'txn_count_24H', 'txn_count_7D',
            'vel_1h_vs_24h', 'vel_6h_vs_7d', 'rapid_fire',
            
            # Amount features
            'amt_mean_24H', 'amt_std_24H', 'amt_mean_7D', 'amt_std_7D', 'amt_mean_30D', 'amt_std_30D',
            'amt_zscore_24h', 'amt_zscore_7d', 'amt_pct_vs_24h', 'amt_pct_vs_7d',
            'is_round_amount', 'is_very_round',
            
            # Merchant features
            'merch_freq_account', 'merch_amt_mean', 'merch_amt_std', 'merch_fraud_rate',
            'merch_cat_freq', 'merch_cat_risk', 'is_new_merchant',
            
            # Account features
            'account_age_days', 'avg_daily_spending', 'merchant_diversity', 'category_diversity',
            
            # Risk features
            'risk_time', 'risk_amount', 'risk_velocity', 'risk_new_merchant', 'risk_score'
        ]
        
        return base_features + advanced_features
