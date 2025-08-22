"""Balance forecasting models using Prophet and SARIMAX."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("Statsmodels not available")

import joblib

class BalanceForecaster:
    """Handles account balance forecasting."""
    
    def __init__(self, forecast_horizon: int = 30):
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.scalers = {}
        
    def prepare_time_series(self, df: pd.DataFrame, account_id: str) -> pd.DataFrame:
        """Prepare time series data for forecasting."""
        
        # Filter for specific account
        account_df = df[df['account_id'] == account_id].copy()
        account_df = account_df.sort_values('timestamp')
        
        # Create daily aggregates
        daily_df = account_df.groupby(account_df['timestamp'].dt.date).agg({
            'balance': 'last',  # End of day balance
            'amount': ['sum', 'count'],  # Daily net flow and transaction count
            'is_debit': 'sum',
            'is_credit': 'sum',
            'is_weekend': 'first',
            'is_holiday': 'first'
        }).reset_index()
        
        # Flatten column names
        daily_df.columns = ['ds', 'balance', 'net_flow', 'txn_count', 'debit_count', 'credit_count', 'is_weekend', 'is_holiday']
        daily_df['ds'] = pd.to_datetime(daily_df['ds'])
        
        # Fill missing dates
        date_range = pd.date_range(start=daily_df['ds'].min(), end=daily_df['ds'].max(), freq='D')
        full_df = pd.DataFrame({'ds': date_range})
        daily_df = full_df.merge(daily_df, on='ds', how='left')
        
        # Forward fill balance, zero fill others
        daily_df['balance'] = daily_df['balance'].fillna(method='ffill')
        daily_df[['net_flow', 'txn_count', 'debit_count', 'credit_count']] = daily_df[['net_flow', 'txn_count', 'debit_count', 'credit_count']].fillna(0)
        daily_df[['is_weekend', 'is_holiday']] = daily_df[['is_weekend', 'is_holiday']].fillna(0)
        
        return daily_df
    
    def train_prophet_model(self, daily_df: pd.DataFrame, account_id: str) -> Optional[object]:
        """Train Prophet model for balance forecasting."""
        
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available")
            return None
        
        if len(daily_df) < 30:  # Need minimum data
            logger.warning(f"Insufficient data for {account_id}: {len(daily_df)} days")
            return None
        
        logger.info(f"Training Prophet model for {account_id}")
        
        # Prepare Prophet format
        prophet_df = daily_df[['ds', 'balance']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Add regressors
        regressor_df = daily_df[['ds', 'is_weekend', 'is_holiday', 'net_flow', 'txn_count']].copy()
        prophet_df = prophet_df.merge(regressor_df, on='ds')
        
        # Initialize Prophet with custom settings
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False
        )
        
        # Add regressors
        model.add_regressor('is_weekend')
        model.add_regressor('is_holiday')
        model.add_regressor('net_flow')
        model.add_regressor('txn_count')
        
        # Fit model
        try:
            model.fit(prophet_df)
            self.models[f'prophet_{account_id}'] = model
            return model
        except Exception as e:
            logger.error(f"Error training Prophet for {account_id}: {e}")
            return None
    
    def train_sarimax_model(self, daily_df: pd.DataFrame, account_id: str) -> Optional[object]:
        """Train SARIMAX model for balance forecasting."""
        
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available")
            return None
        
        if len(daily_df) < 50:  # Need more data for SARIMAX
            logger.warning(f"Insufficient data for SARIMAX {account_id}: {len(daily_df)} days")
            return None
        
        logger.info(f"Training SARIMAX model for {account_id}")
        
        try:
            # Prepare data
            y = daily_df['balance'].values
            exog = daily_df[['is_weekend', 'is_holiday', 'net_flow', 'txn_count']].values
            
            # Fit SARIMAX model
            model = SARIMAX(
                y,
                exog=exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            self.models[f'sarimax_{account_id}'] = fitted_model
            
            return fitted_model
            
        except Exception as e:
            logger.error(f"Error training SARIMAX for {account_id}: {e}")
            return None
    
    def forecast_balance(self, account_id: str, daily_df: pd.DataFrame, model_type: str = 'prophet') -> Optional[pd.DataFrame]:
        """Generate balance forecast for an account."""
        
        model_key = f'{model_type}_{account_id}'
        
        if model_key not in self.models:
            logger.warning(f"No {model_type} model found for {account_id}")
            return None
        
        if model_type == 'prophet' and PROPHET_AVAILABLE:
            return self._forecast_prophet(account_id, daily_df)
        elif model_type == 'sarimax' and STATSMODELS_AVAILABLE:
            return self._forecast_sarimax(account_id, daily_df)
        else:
            logger.warning(f"Model type {model_type} not available")
            return None
    
    def _forecast_prophet(self, account_id: str, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Generate Prophet forecast."""
        
        model = self.models[f'prophet_{account_id}']
        
        # Create future dataframe
        last_date = daily_df['ds'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.forecast_horizon,
            freq='D'
        )
        
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Add regressor values (simplified assumptions)
        future_df['is_weekend'] = (future_df['ds'].dt.dayofweek >= 5).astype(int)
        future_df['is_holiday'] = 0  # Simplified
        future_df['net_flow'] = daily_df['net_flow'].tail(30).mean()  # Use recent average
        future_df['txn_count'] = daily_df['txn_count'].tail(30).mean()
        
        # Generate forecast
        forecast = model.predict(future_df)
        
        # Extract key columns
        result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        result_df.columns = ['date', 'forecast', 'lower_bound', 'upper_bound']
        result_df['account_id'] = account_id
        result_df['model_type'] = 'prophet'
        
        return result_df
    
    def _forecast_sarimax(self, account_id: str, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Generate SARIMAX forecast."""
        
        model = self.models[f'sarimax_{account_id}']
        
        # Prepare exogenous variables for forecast
        exog_future = np.tile(
            daily_df[['is_weekend', 'is_holiday', 'net_flow', 'txn_count']].tail(7).mean().values,
            (self.forecast_horizon, 1)
        )
        
        # Generate forecast
        forecast_result = model.forecast(steps=self.forecast_horizon, exog=exog_future)
        forecast_ci = model.get_forecast(steps=self.forecast_horizon, exog=exog_future).conf_int()
        
        # Create result dataframe
        last_date = daily_df['ds'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=self.forecast_horizon,
            freq='D'
        )
        
        result_df = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast_result,
            'lower_bound': forecast_ci.iloc[:, 0],
            'upper_bound': forecast_ci.iloc[:, 1],
            'account_id': account_id,
            'model_type': 'sarimax'
        })
        
        return result_df
    
    def calculate_forecast_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {}
        
        # Calculate metrics
        mae = np.mean(np.abs(actual_clean - predicted_clean))
        mape = np.mean(np.abs((actual_clean - predicted_clean) / (actual_clean + 1e-6))) * 100
        rmse = np.sqrt(np.mean((actual_clean - predicted_clean) ** 2))
        
        return {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'mean_actual': actual_clean.mean(),
            'mean_predicted': predicted_clean.mean()
        }
    
    def detect_overdraft_risk(self, forecast_df: pd.DataFrame, risk_threshold: float = 0.0) -> pd.DataFrame:
        """Detect overdraft risk from forecasts."""
        
        forecast_df = forecast_df.copy()
        
        # Risk flags
        forecast_df['overdraft_risk'] = (forecast_df['forecast'] < risk_threshold).astype(int)
        forecast_df['high_overdraft_risk'] = (forecast_df['upper_bound'] < risk_threshold).astype(int)
        
        # Days to overdraft
        forecast_df['days_to_overdraft'] = np.nan
        
        for account_id in forecast_df['account_id'].unique():
            account_mask = forecast_df['account_id'] == account_id
            account_forecast = forecast_df[account_mask].copy()
            
            overdraft_dates = account_forecast[account_forecast['overdraft_risk'] == 1]
            if len(overdraft_dates) > 0:
                first_overdraft = overdraft_dates['date'].min()
                days_to = (first_overdraft - datetime.now().date()).days
                forecast_df.loc[account_mask, 'days_to_overdraft'] = max(0, days_to)
        
        return forecast_df

def main():
    """Main function for training forecasting models."""
    
    logging.basicConfig(level=logging.INFO)
    
    from sqlalchemy import create_engine
    
    # Load feature data
    engine = create_engine("sqlite:///data/transactions.db")
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM feature_store", conn)
        
        if len(df) == 0:
            print("No feature data found. Run feature engineering first.")
            return
        
        # Initialize forecaster
        forecaster = BalanceForecaster(forecast_horizon=30)
        
        # Get unique accounts with sufficient data
        account_counts = df['account_id'].value_counts()
        valid_accounts = account_counts[account_counts >= 30].index[:10]  # Top 10 accounts for demo
        
        print(f"Training forecasting models for {len(valid_accounts)} accounts...")
        
        all_forecasts = []
        
        for account_id in valid_accounts:
            print(f"Processing {account_id}...")
            
            # Prepare time series
            daily_df = forecaster.prepare_time_series(df, account_id)
            
            if len(daily_df) < 30:
                continue
            
            # Train models
            if PROPHET_AVAILABLE:
                prophet_model = forecaster.train_prophet_model(daily_df, account_id)
                if prophet_model:
                    prophet_forecast = forecaster.forecast_balance(account_id, daily_df, 'prophet')
                    if prophet_forecast is not None:
                        all_forecasts.append(prophet_forecast)
            
            if STATSMODELS_AVAILABLE:
                sarimax_model = forecaster.train_sarimax_model(daily_df, account_id)
                if sarimax_model:
                    sarimax_forecast = forecaster.forecast_balance(account_id, daily_df, 'sarimax')
                    if sarimax_forecast is not None:
                        all_forecasts.append(sarimax_forecast)
        
        # Combine all forecasts
        if all_forecasts:
            combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
            
            # Detect overdraft risks
            risk_forecasts = forecaster.detect_overdraft_risk(combined_forecasts)
            
            # Save forecasts to database
            with engine.connect() as conn:
                risk_forecasts.to_sql('balance_forecasts', conn, if_exists='replace', index=False)
            
            print(f"\nForecasting completed!")
            print(f"Generated forecasts for {len(valid_accounts)} accounts")
            print(f"Total forecast records: {len(combined_forecasts)}")
            
            # Show overdraft risks
            overdraft_accounts = risk_forecasts[risk_forecasts['overdraft_risk'] == 1]['account_id'].nunique()
            print(f"Accounts with overdraft risk: {overdraft_accounts}")
            
        else:
            print("No forecasts generated. Check data availability and model requirements.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run data loading and feature engineering first.")

if __name__ == "__main__":
    main()
