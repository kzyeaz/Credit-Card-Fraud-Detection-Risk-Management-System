"""Streamlit dashboard for Customer Transaction Intelligence system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.fraud_detection import AnomalyDetector
from src.models.enhanced_anomaly_detection import EnhancedAnomalyDetector
from src.features.advanced_features import AdvancedFeatureEngineer
from explainability.model_explainer import ModelExplainer
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Transaction Intelligence Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load data from database."""
    try:
        engine = create_engine("sqlite:///data/transactions.db")
        
        # Load transactions with features
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM feature_store LIMIT 5000", conn)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)  # Cache for 60 seconds only
def load_forecasts():
    """Load balance forecasts."""
    try:
        engine = create_engine("sqlite:///data/transactions.db")
        
        with engine.connect() as conn:
            forecasts = pd.read_sql("SELECT * FROM balance_forecasts", conn)
            
        return forecasts
    except Exception as e:
        logger.warning(f"No forecast data available: {e}")
        return pd.DataFrame()

def main():
    """Main dashboard function."""
    
    st.title("🔍 Customer Transaction Intelligence Dashboard")
    st.markdown("Real-time fraud detection and balance forecasting system")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Anomaly Detection", "Balance Forecasting", "Model Performance", "Alert Management"]
    )
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data available. Please run the data pipeline first.")
        st.code("""
        # Run these commands to set up data:
        python scripts/download_data.py
        python src/ingestion/data_loader.py
        python src/features/feature_engineering.py
        """)
        return
    
    # Main content based on selected page
    if page == "Overview":
        show_overview(df)
    elif page == "Anomaly Detection":
        show_anomaly_detection(df)
    elif page == "Balance Forecasting":
        show_balance_forecasting(df)
    elif page == "Model Performance":
        show_model_performance(df)
    elif page == "Alert Management":
        show_alert_management(df)

def show_overview(df):
    """Show system overview page."""
    
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    
    with col2:
        fraud_rate = df['is_fraud'].mean() if 'is_fraud' in df.columns else 0
        st.metric("Fraud Rate", f"{fraud_rate:.2%}")
    
    with col3:
        unique_accounts = df['account_id'].nunique()
        st.metric("Active Accounts", f"{unique_accounts:,}")
    
    with col4:
        avg_amount = df['amount'].mean()
        st.metric("Avg Transaction", f"${avg_amount:.2f}")
    
    # Transaction volume over time
    st.subheader("📈 Transaction Volume Trends")
    
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_stats = df.groupby('date').agg({
        'transaction_id': 'count',
        'amount': 'sum',
        'is_fraud': 'sum'
    }).reset_index()
    
    daily_stats.columns = ['date', 'transaction_count', 'total_amount', 'fraud_count']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Transaction Count', 'Daily Fraud Count'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_stats['date'], y=daily_stats['transaction_count'], 
                  name='Transactions', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=daily_stats['date'], y=daily_stats['fraud_count'], 
                  name='Fraud Cases', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Category breakdown
    st.subheader("📊 Transaction Categories")
    
    # Create category mapping from encoded numbers to names
    category_mapping = {
        0: 'Entertainment',
        1: 'Food & Dining', 
        2: 'Gas & Transport',
        3: 'Grocery (POS)',
        4: 'Grocery (Online)',
        5: 'Health & Fitness',
        6: 'Home',
        7: 'Kids & Pets',
        8: 'Miscellaneous (Online)',
        9: 'Miscellaneous (POS)',
        10: 'Personal Care',
        11: 'Shopping (Online)',
        12: 'Shopping (POS)',
        13: 'Travel'
    }
    
    # Check if merchant_category exists, otherwise use category_encoded
    category_col = 'merchant_category' if 'merchant_category' in df.columns else 'category_encoded'
    if category_col in df.columns:
        # Map encoded categories to readable names
        if category_col == 'category_encoded':
            df['category_display'] = df[category_col].map(category_mapping).fillna('Unknown')
            category_col = 'category_display'
        
        category_stats = df.groupby(category_col).agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
    else:
        st.warning("Category data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        count_data = category_stats[('amount', 'count')].reset_index()
        count_data.columns = ['category', 'count']
        fig = px.pie(count_data, values='count', names='category', 
                    title="Transaction Count by Category")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sum_data = category_stats[('amount', 'sum')].reset_index()
        sum_data.columns = ['category', 'amount']
        fig = px.pie(sum_data, values='amount', names='category',
                    title="Transaction Amount by Category")
        st.plotly_chart(fig, use_container_width=True)

def show_anomaly_detection(df):
    """Show anomaly detection page."""
    
    st.header("🚨 Anomaly Detection")
    
    # Model selection
    model_type = st.selectbox(
        "Select Detection Model:",
        ["Enhanced Ensemble", "Basic Isolation Forest"],
        help="Enhanced model uses advanced features and ensemble methods"
    )
    
    # Load or train models
    try:
        if model_type == "Enhanced Ensemble":
            detector = EnhancedAnomalyDetector(contamination=0.1)
            
            # Use existing feature store data directly (already has engineered features)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['transaction_id', 'account_id', 'unix_timestamp', 'is_fraud']
            feature_columns = [col for col in numeric_cols if col not in exclude_cols]
            
            # Always train fresh models to avoid dimension mismatches
            st.info("Training enhanced models with current feature set...")
            
            # Prepare features and labels
            X = detector.prepare_features(df, feature_columns)
            y = df['is_fraud'].values if 'is_fraud' in df.columns else None
            
            # Train models
            detector.train_models(X, y)
            detector.save_models()
            st.success("Enhanced models trained successfully!")
            
            # Get predictions
            predictions, scores = detector.predict_anomalies(X, method='ensemble')
            
        else:
            # Use basic model
            detector = AnomalyDetector()
            
            # Get feature columns dynamically
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['transaction_id', 'account_id', 'timestamp', 'is_fraud', 'unix_timestamp']
            feature_columns = [col for col in numeric_cols if col not in exclude_cols]
            
            # Load or train basic model
            models_loaded = detector.load_models()
            
            if not models_loaded or detector.isolation_forest is None:
                st.info("Training basic models...")
                X = detector.prepare_features(df, feature_columns)
                detector.train_isolation_forest(X)
                detector.save_models()
            else:
                st.success("Basic models loaded successfully!")
            
            # Get predictions
            X = detector.prepare_features(df, feature_columns)
            predictions, scores = detector.predict_anomalies(X, 'isolation_forest')
        
        # Add predictions to dataframe
        df['is_anomaly'] = predictions
        df['anomaly_score'] = scores
        
    except Exception as e:
        st.error(f"Error with anomaly detection: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        st.text("Full error details:")
        st.code(traceback.format_exc())
        return
        
    # Anomaly statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        anomaly_count = predictions.sum()
        st.metric("Anomalies Detected", f"{anomaly_count:,}")
    
    with col2:
        anomaly_rate = predictions.mean()
        st.metric("Anomaly Rate", f"{anomaly_rate:.2%}")
    
    with col3:
        if 'is_fraud' in df.columns:
            precision = (df['is_fraud'] & df['is_anomaly']).sum() / max(df['is_anomaly'].sum(), 1)
            st.metric("Precision", f"{precision:.2%}")
        else:
            st.metric("Precision", "N/A")
    
    # Anomaly score distribution
    st.subheader("📊 Anomaly Score Distribution")
    
    fig = px.histogram(df, x='anomaly_score', nbins=50, 
                      title="Distribution of Anomaly Scores")
    fig.add_vline(x=np.percentile(scores, 95), line_dash="dash", 
                  annotation_text="95th Percentile")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent anomalies table
    st.subheader("🔍 Recent Anomalies")
    
    anomalies_df = df[df['is_anomaly'] == 1].copy()
    if len(anomalies_df) > 0:
        anomalies_df = anomalies_df.sort_values('timestamp', ascending=False)
    
    if len(anomalies_df) > 0:
        # Check available columns and use fallbacks
        display_cols = []
        if 'timestamp' in anomalies_df.columns:
            display_cols.append('timestamp')
        if 'account_id' in anomalies_df.columns:
            display_cols.append('account_id')
        if 'amount' in anomalies_df.columns:
            display_cols.append('amount')
        if 'merchant_name' in anomalies_df.columns:
            display_cols.append('merchant_name')
        elif 'merchant' in anomalies_df.columns:
            display_cols.append('merchant')
        if 'merchant_category' in anomalies_df.columns:
            display_cols.append('merchant_category')
        elif 'category' in anomalies_df.columns:
            display_cols.append('category')
        if 'anomaly_score' in anomalies_df.columns:
            display_cols.append('anomaly_score')
        
        available_cols = display_cols
        
        st.dataframe(
            anomalies_df[available_cols].head(20),
            use_container_width=True
        )
        
        # Detailed view for selected anomaly
        if st.checkbox("Show Detailed Analysis"):
            selected_idx = st.selectbox(
                "Select transaction for detailed analysis:",
                range(min(10, len(anomalies_df))),
                format_func=lambda x: f"{anomalies_df.iloc[x]['transaction_id']} - ${anomalies_df.iloc[x]['amount']:.2f}"
            )
            
            selected_transaction = anomalies_df.iloc[selected_idx]
            
            st.subheader("Transaction Details")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Basic Information:**")
                st.write(f"- Transaction ID: {selected_transaction['transaction_id']}")
                st.write(f"- Account: {selected_transaction['account_id']}")
                st.write(f"- Amount: ${selected_transaction['amount']:.2f}")
                st.write(f"- Merchant: {selected_transaction.get('merchant_name', selected_transaction.get('merchant', 'N/A'))}")
                st.write(f"- Category: {selected_transaction.get('merchant_category', selected_transaction.get('category', 'N/A'))}")
                st.write(f"- Time: {selected_transaction['timestamp']}")
            
            with col2:
                st.write("**Risk Indicators:**")
                st.write(f"- Anomaly Score: {selected_transaction['anomaly_score']:.3f}")
                st.write(f"- Hour of Day: {selected_transaction.get('hour_of_day', 'N/A')}")
                st.write(f"- Weekend: {'Yes' if selected_transaction.get('is_weekend', 0) else 'No'}")
                st.write(f"- New Merchant: {'Yes' if selected_transaction.get('is_new_merchant', 0) else 'No'}")
    else:
        st.info("No anomalies detected in current data.")

def show_balance_forecasting(df):
    """Show balance forecasting page."""
    
    st.header("📈 Balance Forecasting")
    
    # Load forecasts
    forecasts_df = load_forecasts()
    
    if forecasts_df.empty:
        st.warning("No forecast data available. Run forecasting models first.")
        st.code("python src/models/forecasting.py")
        return
    
    # Account selector
    available_accounts = forecasts_df['account_id'].unique()
    selected_account = st.selectbox("Select Account:", available_accounts)
    
    # Filter forecasts for selected account
    account_forecasts = forecasts_df[forecasts_df['account_id'] == selected_account]
    
    # Current balance and risk metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_balance = df[df['account_id'] == selected_account]['balance'].iloc[-1]
        st.metric("Current Balance", f"${current_balance:.2f}")
    
    with col2:
        min_forecast = account_forecasts['forecast'].min()
        st.metric("Minimum Forecast", f"${min_forecast:.2f}")
    
    with col3:
        overdraft_risk = (account_forecasts['overdraft_risk'].sum() > 0)
        st.metric("Overdraft Risk", "Yes" if overdraft_risk else "No")
    
    # Forecast chart
    st.subheader("Balance Forecast")
    
    # Historical balance
    account_history = df[df['account_id'] == selected_account].copy()
    account_history['date'] = pd.to_datetime(account_history['timestamp']).dt.date
    daily_balance = account_history.groupby('date')['balance'].last().reset_index()
    
    # Create forecast visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=daily_balance['date'],
        y=daily_balance['balance'],
        mode='lines',
        name='Historical Balance',
        line=dict(color='blue')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=account_forecasts['date'],
        y=account_forecasts['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='green', dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=account_forecasts['date'],
        y=account_forecasts['upper_bound'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=account_forecasts['date'],
        y=account_forecasts['lower_bound'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Confidence Interval',
        fillcolor='rgba(0,255,0,0.2)'
    ))
    
    # Add overdraft line
    fig.add_hline(y=0, line_dash="dot", line_color="red", 
                  annotation_text="Overdraft Line")
    
    fig.update_layout(
        title=f"Balance Forecast for {selected_account}",
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk analysis
    if overdraft_risk:
        st.subheader("⚠️ Overdraft Risk Analysis")
        
        risk_dates = account_forecasts[account_forecasts['overdraft_risk'] == 1]
        if len(risk_dates) > 0:
            first_risk_date = risk_dates['date'].min()
            days_to_risk = (pd.to_datetime(first_risk_date) - datetime.now()).days
            
            st.warning(f"Potential overdraft in {days_to_risk} days ({first_risk_date})")
            
            # Show risk mitigation suggestions
            st.info("""
            **Recommended Actions:**
            - Monitor large upcoming transactions
            - Consider balance transfer or credit line
            - Set up low balance alerts
            - Review spending patterns
            """)

def show_model_performance(df):
    """Show model performance metrics."""
    
    st.header("📊 Model Performance")
    
    # Model evaluation metrics
    try:
        # Try to load saved metrics first
        import joblib
        if os.path.exists('models/evaluation_metrics.pkl'):
            anomaly_metrics = joblib.load('models/evaluation_metrics.pkl')
            
            st.subheader("🚨 Anomaly Detection Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precision", f"{anomaly_metrics['precision']:.3f}")
            
            with col2:
                st.metric("Recall", f"{anomaly_metrics['recall']:.3f}")
            
            with col3:
                st.metric("F1 Score", f"{anomaly_metrics['f1']:.3f}")
            
            with col4:
                st.metric("AUC", f"{anomaly_metrics['auc']:.3f}")
                
            # Additional metrics
            st.subheader("📈 Detection Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Alert Rate", f"{anomaly_metrics['alert_rate']:.2%}")
            
            with col2:
                st.metric("Fraud Rate", f"{anomaly_metrics['fraud_rate']:.2%}")
        else:
            # Fallback to model evaluator
            from src.evaluation.model_evaluator import ModelEvaluator
            evaluator = ModelEvaluator()
            anomaly_metrics = evaluator.evaluate_anomaly_detection()
            
            if anomaly_metrics:
                st.subheader("🚨 Anomaly Detection Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Precision", f"{anomaly_metrics['precision']:.3f}")
                
                with col2:
                    st.metric("Recall", f"{anomaly_metrics['recall']:.3f}")
                
                with col3:
                    st.metric("F1 Score", f"{anomaly_metrics['f1']:.3f}")
                
                with col4:
                    st.metric("AUC", f"{anomaly_metrics['auc']:.3f}")
            else:
                st.warning("Model evaluation not available. Models may need training.")
            
    except Exception as e:
        st.error(f"Error evaluating models: {e}")
        st.info("Try running: `python fix_model_evaluation.py`")
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    
    try:
        # Mock feature importance for demonstration since SHAP may not be available
        feature_names = ['amount', 'hour', 'day_of_week', 'month', 'amount_zscore', 
                        'merchant_freq', 'category_encoded']
        importance_scores = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.12]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    title="Global Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.info("Feature importance analysis not available. Train explainer first.")

def show_alert_management(df):
    """Show alert management interface."""
    
    st.header("🚨 Alert Management")
    
    # Generate mock alerts for demonstration
    if 'is_fraud' in df.columns:
        alerts_df = df[df['is_fraud'] == 1].copy()
    else:
        # Create mock alerts
        alerts_df = df.sample(min(20, len(df))).copy()
        alerts_df['is_fraud'] = 1
    
    alerts_df['alert_id'] = range(1, len(alerts_df) + 1)
    alerts_df['status'] = np.random.choice(['open', 'investigating', 'closed'], len(alerts_df))
    alerts_df['priority'] = np.random.choice(['high', 'medium', 'low'], len(alerts_df))
    
    # Alert summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        open_alerts = (alerts_df['status'] == 'open').sum()
        st.metric("Open Alerts", open_alerts)
    
    with col2:
        investigating = (alerts_df['status'] == 'investigating').sum()
        st.metric("Investigating", investigating)
    
    with col3:
        high_priority = (alerts_df['priority'] == 'high').sum()
        st.metric("High Priority", high_priority)
    
    with col4:
        avg_amount = alerts_df['amount'].mean()
        st.metric("Avg Alert Amount", f"${avg_amount:.2f}")
    
    # Alerts table
    st.subheader("Alert Queue")
    
    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect("Status", ['open', 'investigating', 'closed'], 
                                     default=['open', 'investigating'])
    with col2:
        priority_filter = st.multiselect("Priority", ['high', 'medium', 'low'], 
                                       default=['high', 'medium', 'low'])
    
    # Apply filters
    filtered_alerts = alerts_df[
        (alerts_df['status'].isin(status_filter)) & 
        (alerts_df['priority'].isin(priority_filter))
    ]
    
    # Display alerts
    display_cols = ['alert_id', 'timestamp', 'account_id', 'amount', 'merchant_name', 
                   'priority', 'status']
    available_cols = [col for col in display_cols if col in filtered_alerts.columns]
    
    st.dataframe(
        filtered_alerts[available_cols].sort_values('timestamp', ascending=False),
        use_container_width=True
    )
    
    # Alert disposition
    if len(filtered_alerts) > 0:
        st.subheader("Alert Actions")
        
        selected_alert = st.selectbox(
            "Select alert to review:",
            filtered_alerts['alert_id'].tolist(),
            format_func=lambda x: f"Alert {x} - ${filtered_alerts[filtered_alerts['alert_id']==x]['amount'].iloc[0]:.2f}"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Mark as False Positive"):
                st.success(f"Alert {selected_alert} marked as false positive")
        
        with col2:
            if st.button("Escalate"):
                st.info(f"Alert {selected_alert} escalated to fraud team")
        
        with col3:
            if st.button("Request More Info"):
                st.info(f"Additional information requested for Alert {selected_alert}")

def show_balance_forecasting(df):
    """Show balance forecasting page."""
    
    st.header("📈 Balance Forecasting")
    
    forecasts_df = load_forecasts()
    
    if forecasts_df.empty:
        st.warning("No forecast data available.")
        return
    
    # Account selector
    available_accounts = forecasts_df['account_id'].unique()
    selected_account = st.selectbox("Select Account:", available_accounts)
    
    account_forecasts = forecasts_df[forecasts_df['account_id'] == selected_account]
    
    # Show forecast visualization (already implemented in show_balance_forecasting above)
    # Current balance and metrics
    account_data = df[df['account_id'] == selected_account]
    # Handle missing balance column
    if 'balance' in account_data.columns:
        current_balance = account_data['balance'].iloc[-1] if len(account_data) > 0 else 0
    else:
        # Calculate balance from transactions if balance column doesn't exist
        current_balance = account_data['amount'].sum() if len(account_data) > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Balance", f"${current_balance:.2f}")
    
    with col2:
        min_forecast = account_forecasts['forecast'].min()
        st.metric("30-Day Min Forecast", f"${min_forecast:.2f}")
    
    with col3:
        overdraft_days = account_forecasts[account_forecasts['overdraft_risk'] == 1]
        if len(overdraft_days) > 0:
            days_to_overdraft = (pd.to_datetime(overdraft_days['date'].min()) - datetime.now()).days
            st.metric("Days to Overdraft Risk", f"{max(0, days_to_overdraft)}")
        else:
            st.metric("Days to Overdraft Risk", "None")

if __name__ == "__main__":
    main()
