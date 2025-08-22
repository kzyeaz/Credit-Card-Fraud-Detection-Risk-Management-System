"""Data ingestion and loading utilities."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import sqlite3
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class TransactionDataLoader:
    """Handles loading and preprocessing of transaction data."""
    
    def __init__(self, db_connection_string: str = "sqlite:///data/transactions.db"):
        self.engine = create_engine(db_connection_string)
        
    def load_raw_transactions(self, file_path: str) -> pd.DataFrame:
        """Load raw transaction data from CSV."""
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Loaded {len(df)} transactions from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading transactions: {e}")
            raise
    
    def create_staging_tables(self):
        """Create staging tables in database."""
        
        # Transactions table
        transactions_sql = """
        CREATE TABLE IF NOT EXISTS transactions_raw (
            transaction_id VARCHAR(50) PRIMARY KEY,
            account_id VARCHAR(50) NOT NULL,
            timestamp DATETIME NOT NULL,
            amount DECIMAL(10,2) NOT NULL,
            merchant_name VARCHAR(200),
            merchant_category VARCHAR(50),
            channel VARCHAR(20),
            city VARCHAR(100),
            is_fraud INTEGER DEFAULT 0,
            balance DECIMAL(12,2),
            V1 DECIMAL(10,6), V2 DECIMAL(10,6), V3 DECIMAL(10,6), V4 DECIMAL(10,6),
            V5 DECIMAL(10,6), V6 DECIMAL(10,6), V7 DECIMAL(10,6), V8 DECIMAL(10,6),
            V9 DECIMAL(10,6), V10 DECIMAL(10,6), V11 DECIMAL(10,6), V12 DECIMAL(10,6),
            V13 DECIMAL(10,6), V14 DECIMAL(10,6), V15 DECIMAL(10,6), V16 DECIMAL(10,6),
            V17 DECIMAL(10,6), V18 DECIMAL(10,6), V19 DECIMAL(10,6), V20 DECIMAL(10,6),
            V21 DECIMAL(10,6), V22 DECIMAL(10,6), V23 DECIMAL(10,6), V24 DECIMAL(10,6),
            V25 DECIMAL(10,6), V26 DECIMAL(10,6), V27 DECIMAL(10,6), V28 DECIMAL(10,6),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Calendar dimension
        calendar_sql = """
        CREATE TABLE IF NOT EXISTS calendar_dim (
            date DATE PRIMARY KEY,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            weekday INTEGER,
            is_weekend INTEGER,
            is_month_end INTEGER,
            is_holiday INTEGER,
            holiday_name VARCHAR(100)
        );
        """
        
        # Account dimension
        account_sql = """
        CREATE TABLE IF NOT EXISTS account_dim (
            account_id VARCHAR(50) PRIMARY KEY,
            segment VARCHAR(20),
            pay_cycle VARCHAR(20),
            created_date DATE,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(transactions_sql))
            conn.execute(text(calendar_sql))
            conn.execute(text(account_sql))
            conn.commit()
            
        logger.info("Created staging tables")
    
    def load_to_database(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """Load DataFrame to database table."""
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Loaded {len(df)} records to {table_name}")
        except Exception as e:
            logger.error(f"Error loading to database: {e}")
            raise
    
    def populate_calendar_dim(self, start_date: str, end_date: str):
        """Populate calendar dimension table."""
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # US Federal Holidays (simplified)
        holidays = {
            '2023-01-01': 'New Year',
            '2023-07-04': 'Independence Day',
            '2023-12-25': 'Christmas',
            '2024-01-01': 'New Year',
            '2024-07-04': 'Independence Day',
            '2024-12-25': 'Christmas'
        }
        
        calendar_data = []
        for date in date_range:
            is_weekend = 1 if date.weekday() >= 5 else 0
            is_month_end = 1 if date == date + pd.offsets.MonthEnd(0) else 0
            date_str = date.strftime('%Y-%m-%d')
            is_holiday = 1 if date_str in holidays else 0
            holiday_name = holidays.get(date_str, None)
            
            calendar_data.append({
                'date': date.date(),
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'weekday': date.weekday(),
                'is_weekend': is_weekend,
                'is_month_end': is_month_end,
                'is_holiday': is_holiday,
                'holiday_name': holiday_name
            })
        
        calendar_df = pd.DataFrame(calendar_data)
        self.load_to_database(calendar_df, 'calendar_dim', if_exists='replace')
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Validate data quality and return metrics."""
        
        quality_metrics = {}
        
        # Missing data rates
        for col in df.columns:
            missing_rate = df[col].isnull().sum() / len(df)
            quality_metrics[f'{col}_missing_rate'] = missing_rate
        
        # Duplicate rate
        duplicate_rate = df.duplicated().sum() / len(df)
        quality_metrics['duplicate_rate'] = duplicate_rate
        
        # Timestamp gaps
        df_sorted = df.sort_values('timestamp')
        time_gaps = df_sorted['timestamp'].diff().dt.total_seconds() / 3600  # hours
        quality_metrics['avg_time_gap_hours'] = time_gaps.mean()
        quality_metrics['max_time_gap_hours'] = time_gaps.max()
        
        # Amount distribution
        quality_metrics['amount_mean'] = df['amount'].mean()
        quality_metrics['amount_std'] = df['amount'].std()
        quality_metrics['negative_amount_rate'] = (df['amount'] < 0).sum() / len(df)
        
        logger.info(f"Data quality metrics: {quality_metrics}")
        return quality_metrics

def main():
    """Main function to run data loading pipeline."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data loader
    loader = TransactionDataLoader()
    
    # Create staging tables
    loader.create_staging_tables()
    
    # Load transaction data
    if Path('data/raw/transactions.csv').exists():
        df = loader.load_raw_transactions('data/raw/transactions.csv')
        
        # Validate data quality
        quality_metrics = loader.validate_data_quality(df)
        
        # Load to database
        loader.load_to_database(df, 'transactions_raw', if_exists='replace')
        
        # Populate calendar dimension
        min_date = df['timestamp'].min().strftime('%Y-%m-%d')
        max_date = df['timestamp'].max().strftime('%Y-%m-%d')
        loader.populate_calendar_dim(min_date, max_date)
        
        # Load account data if exists
        if Path('data/raw/accounts.csv').exists():
            accounts_df = pd.read_csv('data/raw/accounts.csv')
            loader.load_to_database(accounts_df, 'account_dim', if_exists='replace')
        
        print("Data loading completed successfully!")
        print(f"Loaded {len(df)} transactions")
        print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
        
    else:
        print("No transaction data found. Run download_data.py first.")

if __name__ == "__main__":
    main()
