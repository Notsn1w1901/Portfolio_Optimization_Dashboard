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

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252  # Annualized expected return

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

# Advanced Metrics
def sortino_ratio(weights, log_returns, target_return, cov_matrix):
    downside_returns = log_returns[log_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)  # Annualized downside risk
    return (expected_return(weights, log_returns) - target_return) / downside_deviation

def max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def value_at_risk(returns, confidence_level=0.95):
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var

def expected_shortfall(returns, var, confidence_level=0.95):
    tail_losses = returns[returns <= var]
    return tail_losses.mean()

# Streamlit app structure
st.set_page_config(page_title="Portfolio Optimization", layout="wide", initial_sidebar_state="expanded")

# Sidebar Inputs for User Interactivity
st.sidebar.header("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Enter asset tickers (e.g., BBCA.JK, BTC-USD, TSLA)", "BBCA.JK, BTC-USD")
risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)", value=6.0, step=0.1) / 100  # Convert percentage to decimal
investment_amount_idr = st.sidebar.number_input("Investment Amount (IDR)", value=10000000, step=100000)
years_of_data = st.sidebar.number_input("Years of Data", min_value=1, max_value=20, value=5, step=1)
max_weight = st.sidebar.number_input("Maximum weight per asset (%)", min_value=1, max_value=100, value=50) / 100
min_weight = st.sidebar.number_input("Minimum weight per asset (%)", min_value=0, max_value=100, value=0) / 100
usd_price_input = st.sidebar.number_input("USD Price in IDR", value=15000, step=100)  # User input for USD to IDR price

# Define the time period for the data
end_date = datetime.today()
start_date = end_date - timedelta(days=years_of_data * 365)  # Use user input for years

# Download adjusted close price data for each asset
tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]
adj_close_df = pd.DataFrame()

# Fetch data for all tickers entered by the user
for ticker in tickers:
    if ticker:  # Proceed only if the ticker is non-empty
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            adj_close_df[ticker] = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            continue

# Check if any data was fetched
if adj_close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    # Calculate log returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252  # Annualize the covariance matrix

    # Portfolio optimization constraints
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    for i in range(len(tickers)):
        constraints.append({'type': 'ineq', 'fun': lambda weights, i=i: weights[i] - min_weight})
        constraints.append({'type': 'ineq', 'fun': lambda weights, i=i: max_weight - weights[i]})

    bounds = [(min_weight, max_weight)] * len(tickers)
    initial_weights = np.ones(len(tickers)) / len(tickers)
    optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate_input),
                                 method='SLSQP', constraints=constraints, bounds=bounds)

    optimal_weights = optimized_results.x
    capital_allocation_idr = optimal_weights * investment_amount_idr

    # Display optimal weights and capital allocation
    portfolio_df = pd.DataFrame({
        'Asset': tickers,
        'Weight': optimal_weights,
        'Allocated Capital (IDR)': capital_allocation_idr
    })

    # Calculate Portfolio Expected Return and Risk
    portfolio_expected_return = expected_return(optimal_weights, log_returns) * 100
    portfolio_risk = standard_deviation(optimal_weights, cov_matrix) * 100

    # Calculate Advanced Metrics
    target_return = risk_free_rate_input  # Using the risk-free rate as a target return for Sortino ratio
    portfolio_sortino = sortino_ratio(optimal_weights, log_returns, target_return, cov_matrix)
    cumulative_returns = (1 + np.dot(log_returns.values, optimal_weights)).cumprod()
    max_dd = max_drawdown(cumulative_returns)
    portfolio_returns = np.dot(log_returns.values, optimal_weights)
    portfolio_var = value_at_risk(portfolio_returns)
    portfolio_es = expected_shortfall(portfolio_returns, portfolio_var)

    # Display Portfolio Metrics
    st.subheader('Portfolio Metrics')
    st.write(f"📊 **Portfolio Expected Return (Annualized)**: {portfolio_expected_return:.2f}%")
    st.write(f"📉 **Portfolio Risk (Standard Deviation)**: {portfolio_risk:.2f}%")
    st.write(f"📈 **Sortino Ratio**: {portfolio_sortino:.2f}")
    st.write(f"⚠️ **Maximum Drawdown**: {max_dd * 100:.2f}%")
    st.write(f"📉 **Value-at-Risk (VaR) at 95% Confidence**: {portfolio_var * 100:.2f}%")
    st.write(f"📉 **Expected Shortfall (ES) at 95% Confidence**: {portfolio_es * 100:.2f}%")

    # Show optimal portfolio allocation
    st.dataframe(portfolio_df.style.format({
        'Allocated Capital (IDR)': "Rp {:,.2f}", 'Weight': "{:.4f}"
    }))
