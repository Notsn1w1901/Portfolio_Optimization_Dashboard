import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import streamlit as st

# Functions for portfolio statistics
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns, additional_returns=None):
    expected = np.sum(log_returns.mean() * weights[:-1]) * 252  # Annualized expected return
    if additional_returns is not None:
        expected += weights[-1] * additional_returns  # Add mutual fund return
    return expected

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate, additional_returns=None):
    return (expected_return(weights, log_returns, additional_returns) - risk_free_rate) / standard_deviation(weights[:-1], cov_matrix)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate, additional_returns=None):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate, additional_returns)

# Streamlit app configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

st.title("📈 Portfolio Optimization Dashboard")

# Sidebar Inputs
st.sidebar.header("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Stock & Crypto Tickers (e.g., BBCA.JK, BTC-USD)", "BBCA.JK, BTC-USD")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=6.0) / 100
investment_amount = st.sidebar.number_input("Investment Amount (IDR)", value=10000000)
years_of_data = st.sidebar.number_input("Years of Data", min_value=1, max_value=10, value=5)
usd_to_idr = st.sidebar.number_input("USD to IDR Exchange Rate", value=15000)

# Checkbox for Mutual Fund
include_mutual_fund = st.sidebar.checkbox("Include Mutual Fund")
mutual_fund_return = 0.0  # Default
if include_mutual_fund:
    mutual_fund_return = st.sidebar.number_input("Expected Return for Mutual Fund (%)", value=7.0) / 100

# Define the time period
end_date = datetime.today()
start_date = end_date - timedelta(days=years_of_data * 365)

# Fetch stock and crypto data
tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
adj_close_df = pd.DataFrame()

for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data['Adj Close']
    except:
        st.warning(f"⚠️ Could not fetch data for {ticker}")

# Combine stock and crypto data
if adj_close_df.empty:
    st.error("No valid stock/crypto data found. Please check your ticker symbols.")
    st.stop()

# Calculate log returns & covariance matrix
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252

# Add Mutual Fund as a new asset if checked
if include_mutual_fund:
    num_assets = len(adj_close_df.columns) + 1
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 0.5)] * (num_assets - 1) + [(0, 1)]  # Stocks: max 50%, MF: max 100%
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
else:
    num_assets = len(adj_close_df.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 0.5)] * num_assets  # Max 50% per asset
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

# Portfolio Optimization
optimized_results = minimize(
    neg_sharpe_ratio,
    initial_weights,
    args=(log_returns, cov_matrix, risk_free_rate, mutual_fund_return if include_mutual_fund else None),
    method='SLSQP',
    constraints=constraints,
    bounds=bounds
)

optimal_weights = optimized_results.x
capital_allocation_idr = optimal_weights * investment_amount

# Capital Allocation Calculation
current_prices = {ticker: adj_close_df[ticker].iloc[-1] for ticker in adj_close_df.columns}
shares = {}

for i, ticker in enumerate(adj_close_df.columns):
    if '-USD' in ticker:
        shares[ticker] = capital_allocation_idr[i] / usd_to_idr / current_prices[ticker]
    else:
        shares[ticker] = capital_allocation_idr[i] / current_prices[ticker]

# Portfolio Metrics
portfolio_return = expected_return(optimal_weights, log_returns, mutual_fund_return if include_mutual_fund else None) * 100
portfolio_risk = standard_deviation(optimal_weights[:-1] if include_mutual_fund else optimal_weights, cov_matrix) * 100
sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate, mutual_fund_return if include_mutual_fund else None)

# Display Metrics
st.subheader("📊 Portfolio Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Expected Return", f"{portfolio_return:.2f}%")
with col2:
    st.metric("Portfolio Risk (Std Dev)", f"{portfolio_risk:.2f}%")
with col3:
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

# Asset Allocation Table
st.subheader("📝 Asset Allocation")
asset_names = list(adj_close_df.columns) + (["Mutual Fund"] if include_mutual_fund else [])
weights_display = [f"{w * 100:.2f}%" for w in optimal_weights]
capital_display = [f"Rp {alloc:,.2f}" for alloc in capital_allocation_idr]

if include_mutual_fund:
    current_prices["Mutual Fund"] = "N/A"
    shares["Mutual Fund"] = "N/A"

assets_df = pd.DataFrame({
    "Asset": asset_names,
    "Weighting": weights_display,
    "Allocated Capital (IDR)": capital_display,
    "Current Price": [f"Rp {current_prices[ticker]:,.2f}" if ticker in current_prices else "N/A" for ticker in asset_names],
    "Shares": [f"{shares[ticker]:.2f}" if ticker in shares else "N/A" for ticker in asset_names]
})

st.dataframe(assets_df)

# Portfolio Performance Graphs
st.subheader("📈 Portfolio Performance & Allocation")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Portfolio Cumulative Returns")
    cumulative_returns = (1 + np.dot(log_returns.values, optimal_weights[:-1] if include_mutual_fund else optimal_weights)).cumprod()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cumulative_returns, label="Optimized Portfolio", color="green")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Portfolio Weights Distribution")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(optimal_weights, labels=asset_names, autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title("Optimal Portfolio Weights")
    st.pyplot(fig)
