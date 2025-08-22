"""Simplified balance forecasting without external dependencies."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

class SimpleForecaster:
    """Simple balance forecasting using moving averages and trends."""
    
    def __init__(self, forecast_horizon: int = 30):
        self.forecast_horizon = forecast_horizon
    
    def create_balance_forecasts(self, sample_size: int = 5000) -> pd.DataFrame:
        """Create simple balance forecasts for top accounts."""
        
        engine = create_engine("sqlite:///data/transactions.db")
        
        # Get sample data and calculate running balance
        query = f"""
        SELECT account_id, timestamp, amount
        FROM feature_store 
        ORDER BY account_id, timestamp
        LIMIT {sample_size}
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate running balance for each account
        df = df.sort_values(['account_id', 'timestamp'])
        df['balance'] = 0.0
        
        # Initialize starting balances
        account_balances = {}
        for account_id in df['account_id'].unique():
            account_balances[account_id] = np.random.uniform(2000, 8000)
        
        # Calculate running balance
        for idx, row in df.iterrows():
            account_id = row['account_id']
            if account_id not in account_balances:
                account_balances[account_id] = 5000
            
            account_balances[account_id] -= row['amount']  # Subtract spending
            df.at[idx, 'balance'] = account_balances[account_id]
        
        # Get accounts with sufficient transaction history (reduced threshold)
        account_counts = df['account_id'].value_counts()
        top_accounts = account_counts[account_counts >= 30].index[:10]
        
        all_forecasts = []
        
        for account_id in top_accounts:
            account_df = df[df['account_id'] == account_id].copy()
            account_df = account_df.sort_values('timestamp')
            
            # Create daily balance series
            account_df['date'] = account_df['timestamp'].dt.date
            daily_balance = account_df.groupby('date')['balance'].last().reset_index()
            daily_balance['date'] = pd.to_datetime(daily_balance['date'])
            
            if len(daily_balance) < 15:
                continue
            
            # Simple forecasting using trend and seasonality
            forecast = self._simple_forecast(daily_balance, account_id)
            if forecast is not None:
                all_forecasts.append(forecast)
        
        if all_forecasts:
            combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
            
            # Add risk flags
            combined_forecasts['overdraft_risk'] = (combined_forecasts['forecast'] < 0).astype(int)
            combined_forecasts['low_balance_risk'] = (combined_forecasts['forecast'] < 100).astype(int)
            
            return combined_forecasts
        
        return pd.DataFrame()
    
    def _simple_forecast(self, daily_balance: pd.DataFrame, account_id: str) -> pd.DataFrame:
        """Generate simple forecast using moving average and trend."""
        
        # Calculate trend using linear regression on recent data
        recent_data = daily_balance.tail(min(30, len(daily_balance)))
        x = np.arange(len(recent_data))
        y = recent_data['balance'].values
        
        # Simple linear trend
        if len(x) > 1:
            trend = np.polyfit(x, y, 1)[0]
        else:
            trend = 0
        
        # Moving average (adapt window size to available data)
        ma_window = min(7, len(recent_data))
        ma_7 = recent_data['balance'].rolling(ma_window).mean().iloc[-1]
        ma_30_window = min(len(recent_data), 30)
        ma_30 = recent_data['balance'].rolling(ma_30_window).mean().iloc[-1]
        
        # Current balance
        current_balance = daily_balance['balance'].iloc[-1]
        
        # Generate forecasts
        forecast_dates = pd.date_range(
            start=daily_balance['date'].max() + timedelta(days=1),
            periods=self.forecast_horizon,
            freq='D'
        )
        
        forecasts = []
        for i, date in enumerate(forecast_dates):
            # Simple trend projection with some noise
            forecast_value = current_balance + (trend * (i + 1))
            
            # Add weekly seasonality (simplified)
            day_of_week = date.dayofweek
            if day_of_week in [5, 6]:  # Weekend - less spending
                forecast_value += np.random.normal(0, 50)
            else:  # Weekday - more spending
                forecast_value += np.random.normal(-20, 30)
            
            # Add some random variation
            forecast_value += np.random.normal(0, abs(current_balance) * 0.02)
            
            forecasts.append({
                'date': date,
                'forecast': forecast_value,
                'lower_bound': forecast_value - abs(forecast_value) * 0.1,
                'upper_bound': forecast_value + abs(forecast_value) * 0.1,
                'account_id': account_id,
                'model_type': 'simple_trend'
            })
        
        return pd.DataFrame(forecasts)

def main():
    """Main function for simple forecasting."""
    
    logging.basicConfig(level=logging.INFO)
    
    forecaster = SimpleForecaster()
    forecasts_df = forecaster.create_balance_forecasts()
    
    if len(forecasts_df) > 0:
        # Save to database
        engine = create_engine("sqlite:///data/transactions.db")
        with engine.connect() as conn:
            forecasts_df.to_sql('balance_forecasts', conn, if_exists='replace', index=False)
        
        print(f"Forecasting completed!")
        print(f"Generated forecasts for {forecasts_df['account_id'].nunique()} accounts")
        print(f"Total forecast records: {len(forecasts_df)}")
        
        # Show risk summary
        overdraft_accounts = forecasts_df[forecasts_df['overdraft_risk'] == 1]['account_id'].nunique()
        print(f"Accounts with overdraft risk: {overdraft_accounts}")
        
    else:
        print("No forecasts generated")

if __name__ == "__main__":
    main()
