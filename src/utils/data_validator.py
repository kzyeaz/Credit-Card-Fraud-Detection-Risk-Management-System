"""Data validation utilities for the transaction intelligence system."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data quality and consistency."""
    
    def __init__(self):
        self.validation_rules = {
            'required_columns': [
                'transaction_id', 'account_id', 'timestamp', 'amount', 
                'merchant_name', 'merchant_category'
            ],
            'max_missing_rate': 0.1,
            'max_duplicate_rate': 0.05,
            'amount_bounds': (-10000, 10000),
            'timestamp_range_days': 365 * 3  # 3 years max
        }
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate DataFrame schema."""
        
        results = {}
        
        # Check required columns
        missing_cols = set(self.validation_rules['required_columns']) - set(df.columns)
        results['has_required_columns'] = len(missing_cols) == 0
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'timestamp' in df.columns:
            results['timestamp_is_datetime'] = pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        
        if 'amount' in df.columns:
            results['amount_is_numeric'] = pd.api.types.is_numeric_dtype(df['amount'])
        
        return results
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Validate data quality metrics."""
        
        quality_metrics = {}
        
        # Missing data rates
        for col in df.columns:
            missing_rate = df[col].isnull().sum() / len(df)
            quality_metrics[f'{col}_missing_rate'] = missing_rate
            
            if missing_rate > self.validation_rules['max_missing_rate']:
                logger.warning(f"High missing rate in {col}: {missing_rate:.2%}")
        
        # Duplicate rate
        duplicate_rate = df.duplicated().sum() / len(df)
        quality_metrics['duplicate_rate'] = duplicate_rate
        
        if duplicate_rate > self.validation_rules['max_duplicate_rate']:
            logger.warning(f"High duplicate rate: {duplicate_rate:.2%}")
        
        # Amount validation
        if 'amount' in df.columns:
            min_bound, max_bound = self.validation_rules['amount_bounds']
            out_of_bounds = ((df['amount'] < min_bound) | (df['amount'] > max_bound)).sum()
            quality_metrics['amount_out_of_bounds_rate'] = out_of_bounds / len(df)
            
            quality_metrics['amount_zero_rate'] = (df['amount'] == 0).sum() / len(df)
        
        # Timestamp validation
        if 'timestamp' in df.columns:
            now = datetime.now()
            future_txns = (df['timestamp'] > now).sum()
            quality_metrics['future_transactions_rate'] = future_txns / len(df)
            
            old_threshold = now - timedelta(days=self.validation_rules['timestamp_range_days'])
            old_txns = (df['timestamp'] < old_threshold).sum()
            quality_metrics['very_old_transactions_rate'] = old_txns / len(df)
        
        return quality_metrics
    
    def validate_business_rules(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate business logic rules."""
        
        results = {}
        
        # Fraud rate should be reasonable (0.1% - 10%)
        if 'is_fraud' in df.columns:
            fraud_rate = df['is_fraud'].mean()
            results['fraud_rate_reasonable'] = 0.001 <= fraud_rate <= 0.1
            
            if not results['fraud_rate_reasonable']:
                logger.warning(f"Unusual fraud rate: {fraud_rate:.2%}")
        
        # Account should have multiple transactions
        if 'account_id' in df.columns:
            txn_per_account = df['account_id'].value_counts()
            single_txn_accounts = (txn_per_account == 1).sum()
            results['reasonable_account_activity'] = single_txn_accounts / len(txn_per_account) < 0.5
        
        # Balance consistency (if available)
        if 'balance' in df.columns and 'amount' in df.columns:
            # Check for reasonable balance changes
            df_sorted = df.sort_values(['account_id', 'timestamp'])
            df_sorted['balance_change'] = df_sorted.groupby('account_id')['balance'].diff()
            df_sorted['expected_change'] = -df_sorted['amount']  # Negative because spending reduces balance
            
            # Allow for some tolerance due to other transactions
            balance_consistency = np.abs(df_sorted['balance_change'] - df_sorted['expected_change']).fillna(0)
            results['balance_consistency'] = (balance_consistency < 1000).mean() > 0.8
        
        return results
    
    def generate_validation_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive validation report."""
        
        schema_results = self.validate_schema(df)
        quality_metrics = self.validate_data_quality(df)
        business_results = self.validate_business_rules(df)
        
        report = "# Data Validation Report\n\n"
        
        # Schema validation
        report += "## Schema Validation\n"
        for check, passed in schema_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report += f"- {check}: {status}\n"
        
        # Data quality
        report += "\n## Data Quality Metrics\n"
        for metric, value in quality_metrics.items():
            if 'rate' in metric:
                report += f"- {metric}: {value:.2%}\n"
            else:
                report += f"- {metric}: {value:.3f}\n"
        
        # Business rules
        report += "\n## Business Rule Validation\n"
        for check, passed in business_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            report += f"- {check}: {status}\n"
        
        # Overall assessment
        all_schema_pass = all(schema_results.values())
        critical_quality_pass = (
            quality_metrics.get('duplicate_rate', 0) < 0.05 and
            quality_metrics.get('transaction_id_missing_rate', 0) < 0.01
        )
        all_business_pass = all(business_results.values())
        
        report += "\n## Overall Assessment\n"
        if all_schema_pass and critical_quality_pass and all_business_pass:
            report += "✅ **Data is ready for processing**\n"
        else:
            report += "⚠️ **Data has issues that should be addressed**\n"
        
        return report

def main():
    """Main function for data validation."""
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data for validation
    try:
        df = pd.read_csv('data/raw/transactions.csv')
        
        validator = DataValidator()
        report = validator.generate_validation_report(df)
        
        print(report)
        
        # Save report
        with open('data/validation_report.md', 'w') as f:
            f.write(report)
        
        print("\nValidation report saved to data/validation_report.md")
        
    except FileNotFoundError:
        print("No transaction data found. Run download_data.py first.")
    except Exception as e:
        print(f"Validation error: {e}")

if __name__ == "__main__":
    main()
