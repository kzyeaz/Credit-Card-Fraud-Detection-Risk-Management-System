"""Feature engineering pipeline for transaction intelligence."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering for transaction data."""
    
    def __init__(self, db_connection_string: str = "sqlite:///data/transactions.db"):
        self.engine = create_engine(db_connection_string)
        self.rolling_windows = [1, 7, 14, 30]
    
    def load_transactions(self, account_id: Optional[str] = None) -> pd.DataFrame:
        """Load transactions from database."""
        
        query = """
        SELECT t.*, c.is_weekend, c.is_holiday, c.holiday_name, a.segment, a.pay_cycle
        FROM transactions_raw t
        LEFT JOIN calendar_dim c ON DATE(t.timestamp) = c.date
        LEFT JOIN account_dim a ON t.account_id = a.account_id
        """
        
        if account_id:
            query += f" WHERE t.account_id = '{account_id}'"
        
        query += " ORDER BY t.account_id, t.timestamp"
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        
        df = df.copy()
        
        # Basic time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        
        # Time patterns
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        df['is_weekend'] = df['is_weekend'].fillna(0).astype(int)
        df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
        
        # End of month/quarter flags
        df['is_month_end_week'] = (df['day_of_month'] >= 25).astype(int)
        df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)
        
        return df
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features."""
        
        df = df.copy()
        
        # Amount characteristics
        df['amount_abs'] = df['amount'].abs()
        df['is_debit'] = (df['amount'] > 0).astype(int)
        df['is_credit'] = (df['amount'] < 0).astype(int)
        df['amount_log'] = np.log1p(df['amount_abs'])
        
        # Amount categories
        df['amount_category'] = pd.cut(
            df['amount_abs'], 
            bins=[0, 25, 100, 500, 2000, np.inf],
            labels=['micro', 'small', 'medium', 'large', 'xlarge']
        )
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features per account."""
        
        df = df.copy()
        df = df.sort_values(['account_id', 'timestamp'])
        df = df.set_index('timestamp')
        
        # Initialize feature columns
        feature_cols = []
        
        for window in self.rolling_windows:
            window_str = f"{window}d"
            
            # Rolling aggregations using groupby with proper syntax
            df[f'txn_count_{window_str}'] = df.groupby('account_id')['amount'].rolling(f'{window}D').count().reset_index(level=0, drop=True)
            
            # Separate debits and credits
            debit_amounts = df['amount'].where(df['amount'] > 0, 0)
            credit_amounts = df['amount'].where(df['amount'] < 0, 0)
            
            df[f'debit_sum_{window_str}'] = debit_amounts.groupby(df['account_id']).rolling(f'{window}D').sum().reset_index(level=0, drop=True)
            df[f'credit_sum_{window_str}'] = credit_amounts.groupby(df['account_id']).rolling(f'{window}D').sum().reset_index(level=0, drop=True)
            
            # Amount statistics
            df[f'amount_mean_{window_str}'] = df.groupby('account_id')['amount_abs'].rolling(f'{window}D').mean().reset_index(level=0, drop=True)
            df[f'amount_std_{window_str}'] = df.groupby('account_id')['amount_abs'].rolling(f'{window}D').std().reset_index(level=0, drop=True)
            df[f'amount_max_{window_str}'] = df.groupby('account_id')['amount_abs'].rolling(f'{window}D').max().reset_index(level=0, drop=True)
            
            feature_cols.extend([
                f'txn_count_{window_str}', f'debit_sum_{window_str}', f'credit_sum_{window_str}',
                f'amount_mean_{window_str}', f'amount_std_{window_str}', f'amount_max_{window_str}'
            ])
        
        # Reset index
        df = df.reset_index()
        
        # Velocity features (ratios between windows)
        df['velocity_7d_vs_30d'] = df['debit_sum_7d'] / (df['debit_sum_30d'] + 1e-6)
        df['velocity_1d_vs_7d'] = df['debit_sum_1d'] / (df['debit_sum_7d'] + 1e-6)
        df['txn_velocity_7d_vs_30d'] = df['txn_count_7d'] / (df['txn_count_30d'] + 1e-6)
        
        feature_cols.extend(['velocity_7d_vs_30d', 'velocity_1d_vs_7d', 'txn_velocity_7d_vs_30d'])
        
        # Fill NaN values
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        return df
    
    def create_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create merchant and category diversity features."""
        
        df = df.copy()
        df = df.sort_values(['account_id', 'timestamp'])
        df = df.set_index('timestamp')
        
        # Merchant diversity in rolling windows
        for window in [7, 30]:
            window_str = f"{window}d"
            
            # Unique merchants
            df[f'unique_merchants_{window_str}'] = (
                df.groupby('account_id')['merchant_name']
                .rolling(f'{window}D')
                .apply(lambda x: x.nunique())
                .reset_index(level=0, drop=True)
                .fillna(0)
            )
            
            # Unique categories
            df[f'unique_categories_{window_str}'] = (
                df.groupby('account_id')['merchant_category']
                .rolling(f'{window}D')
                .apply(lambda x: x.nunique())
                .reset_index(level=0, drop=True)
                .fillna(0)
            )
        
        # Reset index for merchant flag calculation
        df_temp = df.reset_index()
        
        # New merchant flag
        df_temp['is_new_merchant'] = (
            df_temp.groupby(['account_id', 'merchant_name'])
            .cumcount() == 0
        ).astype(int)
        
        df['is_new_merchant'] = df_temp.set_index('timestamp')['is_new_merchant']
        
        # Category concentration (entropy) - simplified
        def calculate_entropy(series):
            if len(series) == 0:
                return 0
            value_counts = series.value_counts(normalize=True)
            return -sum(value_counts * np.log2(value_counts + 1e-10))
        
        df['category_entropy_7d'] = (
            df.groupby('account_id')['merchant_category']
            .rolling('7D')
            .apply(calculate_entropy)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        
        # Reset index
        df = df.reset_index()
        
        return df
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features."""
        
        df = df.copy()
        df = df.sort_values(['account_id', 'timestamp'])
        
        # Time since last transaction
        df['time_since_last_txn'] = (
            df.groupby('account_id')['timestamp']
            .diff()
            .dt.total_seconds() / 3600  # Convert to hours
        ).fillna(0)
        
        # Transaction burstiness - simplified calculation
        df['txn_count_1h'] = 1  # Each transaction counts as 1 in its hour
        
        # Calculate hourly transaction counts per account
        df['hour_bucket'] = df['timestamp'].dt.floor('H')
        hourly_counts = df.groupby(['account_id', 'hour_bucket']).size().reset_index(name='hourly_txn_count')
        df = df.merge(hourly_counts, on=['account_id', 'hour_bucket'], how='left')
        df['txn_count_1h'] = df['hourly_txn_count'].fillna(1)
        
        # Weekend vs weekday spending patterns
        weekend_spending = df.groupby(['account_id', 'is_weekend'])['amount'].sum().unstack(fill_value=0)
        if 1 in weekend_spending.columns and 0 in weekend_spending.columns:
            weekend_ratio = weekend_spending[1] / (weekend_spending[0] + 1e-6)
            df = df.merge(
                weekend_ratio.rename('weekend_spending_ratio').reset_index(),
                on='account_id',
                how='left'
            )
        else:
            df['weekend_spending_ratio'] = 0
        
        # Channel diversity
        df['channel_diversity_7d'] = (
            df.groupby('account_id')['channel']
            .rolling('7D', on='timestamp')
            .apply(lambda x: x.nunique())
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        
        return df
    
    def detect_payday_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect payday patterns and create related features."""
        
        df = df.copy()
        df = df.sort_values(['account_id', 'timestamp'])
        
        # Identify potential paydays (large credits)
        df['is_potential_payday'] = (
            (df['amount'] < -500) &  # Large credit
            (df['merchant_category'].isin(['other', 'utilities']) | df['merchant_category'].isnull())
        ).astype(int)
        
        # Days since last payday
        df['days_since_payday'] = np.nan
        
        for account_id in df['account_id'].unique():
            account_mask = df['account_id'] == account_id
            account_df = df[account_mask].copy()
            
            payday_dates = account_df[account_df['is_potential_payday'] == 1]['timestamp']
            
            if len(payday_dates) > 0:
                for idx in account_df.index:
                    txn_date = account_df.loc[idx, 'timestamp']
                    days_since = (txn_date - payday_dates[payday_dates <= txn_date]).dt.days
                    if len(days_since) > 0:
                        df.loc[idx, 'days_since_payday'] = days_since.min()
        
        df['days_since_payday'] = df['days_since_payday'].fillna(999)
        
        # Post-payday spending surge flag
        df['is_post_payday'] = (df['days_since_payday'] <= 3).astype(int)
        
        return df
    
    def create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create balance-related features."""
        
        df = df.copy()
        df = df.sort_values(['account_id', 'timestamp'])
        
        # Balance change
        df['balance_change'] = df.groupby('account_id')['balance'].diff().fillna(0)
        
        # Low balance flags
        df['is_low_balance'] = (df['balance'] < 100).astype(int)
        df['is_negative_balance'] = (df['balance'] < 0).astype(int)
        
        # Balance trend
        df['balance_trend_7d'] = (
            df.groupby('account_id')['balance']
            .rolling('7D', on='timestamp')
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            .reset_index(level=0, drop=True)
            .fillna(0)
        )
        
        return df
    
    def create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specifically for anomaly detection."""
        
        df = df.copy()
        
        # Z-scores for amount relative to personal history
        for window in [7, 30]:
            mean_col = f'amount_mean_{window}d'
            std_col = f'amount_std_{window}d'
            
            df[f'amount_zscore_{window}d'] = (
                (df['amount_abs'] - df[mean_col]) / (df[std_col] + 1e-6)
            )
        
        # Unusual timing flags
        df['is_unusual_hour'] = (
            (df['hour_of_day'] < 6) | (df['hour_of_day'] > 23)
        ).astype(int)
        
        # Geographic anomalies (simplified - same city check)
        df['prev_city'] = df.groupby('account_id')['city'].shift(1)
        df['is_different_city'] = (
            (df['city'] != df['prev_city']) & df['prev_city'].notna()
        ).astype(int)
        
        # Rapid successive transactions
        df['rapid_succession'] = (df['time_since_last_txn'] < 0.5).astype(int)  # < 30 minutes
        
        return df
    
    def engineer_features(self, account_id: Optional[str] = None) -> pd.DataFrame:
        """Main feature engineering pipeline."""
        
        logger.info(f"Starting feature engineering for account: {account_id or 'all accounts'}")
        
        # Load data
        df = self.load_transactions(account_id)
        
        if len(df) == 0:
            logger.warning("No transactions found")
            return pd.DataFrame()
        
        # Apply feature engineering steps
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_rolling_features(df)
        df = self.create_merchant_features(df)
        df = self.create_behavioral_features(df)
        df = self.detect_payday_patterns(df)
        df = self.create_balance_features(df)
        df = self.create_anomaly_features(df)
        
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        
        return df
    
    def save_features(self, df: pd.DataFrame, table_name: str = 'feature_store'):
        """Save engineered features to database."""
        
        try:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Saved {len(df)} feature records to {table_name}")
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise
    
    def get_feature_columns(self) -> List[str]:
        """Get list of engineered feature columns."""
        
        base_features = [
            'hour_of_day', 'day_of_week', 'month', 'day_of_month',
            'is_business_hours', 'is_night', 'is_weekend', 'is_holiday',
            'is_month_end_week', 'is_quarter_end',
            'amount_abs', 'is_debit', 'is_credit', 'amount_log',
            'time_since_last_txn', 'txn_count_1h', 'weekend_spending_ratio',
            'channel_diversity_7d', 'days_since_payday', 'is_post_payday',
            'balance_change', 'is_low_balance', 'is_negative_balance', 'balance_trend_7d',
            'is_new_merchant', 'category_entropy_7d', 'is_unusual_hour',
            'is_different_city', 'rapid_succession'
        ]
        
        # Add rolling features
        rolling_features = []
        for window in self.rolling_windows:
            window_str = f"{window}d"
            rolling_features.extend([
                f'txn_count_{window_str}', f'debit_sum_{window_str}', f'credit_sum_{window_str}',
                f'amount_mean_{window_str}', f'amount_std_{window_str}', f'amount_max_{window_str}',
                f'unique_merchants_{window_str}', f'unique_categories_{window_str}',
                f'amount_zscore_{window_str}'
            ])
        
        # Add velocity features
        velocity_features = ['velocity_7d_vs_30d', 'velocity_1d_vs_7d', 'txn_velocity_7d_vs_30d']
        
        # Add V features (anonymized features from dataset)
        v_features = [f'V{i}' for i in range(1, 29)]
        
        return base_features + rolling_features + velocity_features + v_features

def main():
    """Main function to run feature engineering."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Engineer features for all accounts
    features_df = fe.engineer_features()
    
    if len(features_df) > 0:
        # Save features
        fe.save_features(features_df)
        
        print(f"Feature engineering completed!")
        print(f"Generated {len(features_df)} feature records")
        print(f"Feature columns: {len(fe.get_feature_columns())}")
        
        # Show sample statistics
        print("\nFeature Statistics:")
        feature_cols = fe.get_feature_columns()
        available_cols = [col for col in feature_cols if col in features_df.columns]
        print(features_df[available_cols].describe())
        
    else:
        print("No data available for feature engineering")

if __name__ == "__main__":
    main()
