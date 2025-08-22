"""Download and prepare synthetic fraud detection dataset."""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from pathlib import Path

def create_synthetic_dataset():
    """Create a synthetic credit card transaction dataset for testing."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 10,000 transactions over 6 months
    n_transactions = 10000
    n_accounts = 500
    
    # Date range: last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # Generate base data
    data = []
    
    for i in range(n_transactions):
        account_id = np.random.randint(1, n_accounts + 1)
        
        # Generate timestamp with some patterns
        days_offset = np.random.randint(0, 180)
        hour = np.random.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
                               p=[0.05, 0.08, 0.1, 0.12, 0.15, 0.1, 0.08, 0.08, 0.08, 0.08, 0.05, 0.02, 0.01])
        timestamp = start_date + timedelta(days=days_offset, hours=int(hour), 
                                         minutes=int(np.random.randint(0, 60)))
        
        # Generate transaction amount (mostly small, some large)
        if np.random.random() < 0.8:
            amount = np.random.exponential(50)  # Most transactions small
        else:
            amount = np.random.exponential(200)  # Some larger transactions
        
        # Make some transactions credits (positive)
        if np.random.random() < 0.15:  # 15% are credits
            amount = amount * -1
        
        # Merchant categories
        categories = ['grocery', 'gas', 'restaurant', 'retail', 'online', 'entertainment', 
                     'healthcare', 'utilities', 'travel', 'other']
        merchant_category = np.random.choice(categories)
        
        # Channels
        channels = ['online', 'pos', 'atm', 'mobile']
        channel = np.random.choice(channels, p=[0.4, 0.35, 0.15, 0.1])
        
        # Cities (simplified)
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        city = np.random.choice(cities)
        
        # Generate fraud labels (5% fraud rate)
        is_fraud = 0
        if np.random.random() < 0.05:
            is_fraud = 1
            # Fraudulent transactions tend to be larger and at odd hours
            if np.random.random() < 0.7:
                amount *= np.random.uniform(2, 5)
            if np.random.random() < 0.3:
                hour = np.random.choice([2, 3, 4, 5, 23, 0, 1])
        
        # Generate some anonymized features (like Kaggle dataset)
        v_features = {f'V{i}': np.random.normal(0, 1) for i in range(1, 29)}
        
        transaction = {
            'transaction_id': f'txn_{i:06d}',
            'account_id': f'acc_{account_id:04d}',
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_name': f'Merchant_{np.random.randint(1, 1000):03d}',
            'merchant_category': merchant_category,
            'channel': channel,
            'city': city,
            'is_fraud': is_fraud,
            **v_features
        }
        
        data.append(transaction)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Generate account balances
    account_balances = {}
    df['balance'] = 0.0
    
    for idx, row in df.iterrows():
        account_id = row['account_id']
        if account_id not in account_balances:
            account_balances[account_id] = np.random.uniform(1000, 5000)  # Starting balance
        
        # Update balance
        account_balances[account_id] -= row['amount']
        df.at[idx, 'balance'] = account_balances[account_id]
        
        # Simulate periodic deposits (payday)
        if np.random.random() < 0.05:  # 5% chance of deposit
            deposit = np.random.uniform(1000, 3000)
            account_balances[account_id] += deposit
    
    return df

def download_kaggle_dataset():
    """Download Kaggle credit card fraud dataset if available."""
    try:
        import kaggle
        
        # Download the popular credit card fraud dataset
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path='data/raw/',
            unzip=True
        )
        print("Downloaded Kaggle credit card fraud dataset")
        return True
    except Exception as e:
        print(f"Could not download Kaggle dataset: {e}")
        return False

def load_kaggle_dataset():
    """Load and process the Kaggle fraud dataset from archive folder."""
    
    train_path = 'data/archive (17)/fraudTrain.csv'
    test_path = 'data/archive (17)/fraudTest.csv'
    
    if not os.path.exists(train_path):
        return None
    
    print("Loading Kaggle fraud detection dataset...")
    
    # Load a sample of the training data for faster processing
    df_train = pd.read_csv(train_path, nrows=50000)  # Load first 50k rows for demo
    print(f"Loaded training data sample: {len(df_train)} transactions")
    
    # Load test data sample if available
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path, nrows=10000)  # Load first 10k test rows
        print(f"Loaded test data sample: {len(df_test)} transactions")
        
        # Combine datasets
        df = pd.concat([df_train, df_test], ignore_index=True)
    else:
        df = df_train
    
    # Rename columns to match our schema
    column_mapping = {
        'trans_date_trans_time': 'timestamp',
        'cc_num': 'account_id',
        'merchant': 'merchant_name',
        'category': 'merchant_category',
        'amt': 'amount',
        'unix_time': 'unix_timestamp'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create transaction_id
    df['transaction_id'] = 'txn_' + df.index.astype(str).str.zfill(8)
    
    # Convert account_id to string format
    df['account_id'] = 'acc_' + df['account_id'].astype(str)
    
    # Add channel information (inferred from category)
    df['channel'] = df['merchant_category'].map({
        'grocery_pos': 'pos',
        'gas_transport': 'pos', 
        'misc_pos': 'pos',
        'grocery_net': 'online',
        'misc_net': 'online',
        'shopping_net': 'online',
        'shopping_pos': 'pos'
    }).fillna('pos')
    
    # Generate synthetic balance data efficiently
    print("Generating balance information...")
    
    # Sort by account and timestamp for efficient processing
    df = df.sort_values(['account_id', 'timestamp']).reset_index(drop=True)
    
    # Initialize balances for unique accounts
    unique_accounts = df['account_id'].unique()
    account_balances = {acc: np.random.uniform(2000, 10000) for acc in unique_accounts}
    
    # Vectorized balance calculation
    df['balance'] = 0.0
    current_account = None
    current_balance = 0
    
    for idx, row in df.iterrows():
        if row['account_id'] != current_account:
            current_account = row['account_id']
            current_balance = account_balances[current_account]
        
        # Update balance
        current_balance -= row['amount']
        df.at[idx, 'balance'] = current_balance
        
        # Periodic deposits (simplified)
        if idx % 500 == 0 and np.random.random() < 0.3:
            deposit = np.random.uniform(2000, 5000)
            current_balance += deposit
    
    return df

def main():
    """Main function to download and prepare data."""
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Try to load Kaggle dataset from archive first
    df = load_kaggle_dataset()
    
    if df is not None:
        # Save processed dataset
        df.to_csv('data/raw/transactions.csv', index=False)
        print(f"Processed Kaggle dataset: {len(df)} transactions")
        print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Unique accounts: {df['account_id'].nunique()}")
        
        # Create account metadata from the dataset
        account_info = df.groupby('account_id').agg({
            'first': 'first',
            'last': 'first', 
            'gender': 'first',
            'job': 'first',
            'dob': 'first',
            'timestamp': 'min'
        }).reset_index()
        
        account_info['segment'] = np.random.choice(['premium', 'standard', 'basic'], 
                                                 len(account_info), p=[0.2, 0.5, 0.3])
        account_info['pay_cycle'] = np.random.choice(['weekly', 'biweekly', 'monthly'], 
                                                   len(account_info), p=[0.1, 0.3, 0.6])
        account_info['created_date'] = account_info['timestamp']
        
        # Save account metadata
        account_cols = ['account_id', 'segment', 'pay_cycle', 'created_date']
        account_info[account_cols].to_csv('data/raw/accounts.csv', index=False)
        print("Created account metadata from dataset")
        
    else:
        print("Kaggle dataset not found in archive, creating synthetic dataset...")
        # Fallback to synthetic data
        df = create_synthetic_dataset()
        df.to_csv('data/raw/transactions.csv', index=False)
        print(f"Created synthetic dataset with {len(df)} transactions")
        
        # Create sample account metadata
        accounts_df = pd.DataFrame({
            'account_id': [f'acc_{i:04d}' for i in range(1, 501)],
            'segment': np.random.choice(['premium', 'standard', 'basic'], 500, p=[0.2, 0.5, 0.3]),
            'pay_cycle': np.random.choice(['weekly', 'biweekly', 'monthly'], 500, p=[0.1, 0.3, 0.6]),
            'created_date': pd.date_range(start='2020-01-01', end='2023-01-01', periods=500)
        })
        accounts_df.to_csv('data/raw/accounts.csv', index=False)
        print("Created account metadata")

if __name__ == "__main__":
    main()
