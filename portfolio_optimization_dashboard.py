import yfinance as yf
import investpy  # For mutual fund data
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

# Streamlit app configuration
st.set_page_config(page_title="Portfolio Optimization", layout="wide")

st.title("📈 Portfolio Optimization Dashboard")

# Sidebar Inputs
st.sidebar.header("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Stock & Crypto Tickers (e.g., BBCA.JK, BTC-USD)", "BBCA.JK, BTC-USD")
mutual_funds_input = st.sidebar.text_input("Mutual Fund Codes (e.g., RD1, RD2)", "")  # New mutual fund input
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", value=6.0) / 100
investment_amount = st.sidebar.number_input("Investment Amount (IDR)", value=10000000)
years_of_data = st.sidebar.number_input("Years of Data", min_value=1, max_value=10, value=5)
usd_to_idr = st.sidebar.number_input("USD to IDR Exchange Rate", value=15000)

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

# Fetch mutual fund data
mutual_funds = [mf.strip() for mf in mutual_funds_input.split(',') if mf.strip()]
mf_prices = pd.DataFrame()

for mf in mutual_funds:
    try:
        mf_data = investpy.funds.get_fund_historical_data(fund=mf, country="indonesia", from_date=start_date.strftime("%d/%m/%Y"), to_date=end_date.strftime("%d/%m/%Y"))
        mf_prices[mf] = mf_data["Close"]
    except:
        st.warning(f"⚠️ Could not fetch mutual fund data for {mf}")

# Combine stock, crypto, and mutual fund data
if not adj_close_df.empty and not mf_prices.empty:
    combined_df = pd.concat([adj_close_df, mf_prices], axis=1).dropna()
elif not adj_close_df.empty:
    combined_df = adj_close_df.dropna()
elif not mf_prices.empty:
    combined_df = mf_prices.dropna()
else:
    st.error("No valid data found. Please check your ticker symbols.")
    st.stop()

# Calculate log returns & covariance matrix
log_returns = np.log(combined_df / combined_df.shift(1)).dropna()
cov_matrix = log_returns.cov() * 252

# Portfolio Optimization
num_assets = len(combined_df.columns)
initial_weights = np.ones(num_assets) / num_assets
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.5)] * num_assets  # Max 50% per asset

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate),
                             method='SLSQP', constraints=constraints, bounds=bounds)

optimal_weights = optimized_results.x
capital_allocation_idr = optimal_weights * investment_amount

# Capital Allocation Calculation
current_prices = {ticker: combined_df[ticker].iloc[-1] for ticker in combined_df.columns}
shares = {}

for i, ticker in enumerate(combined_df.columns):
    if '-USD' in ticker:
        shares[ticker] = capital_allocation_idr[i] / usd_to_idr / current_prices[ticker]
    else:
        shares[ticker] = capital_allocation_idr[i] / current_prices[ticker]

# Portfolio Metrics
portfolio_return = expected_return(optimal_weights, log_returns) * 100
portfolio_risk = standard_deviation(optimal_weights, cov_matrix) * 100
sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

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
assets_df = pd.DataFrame({
    "Asset": combined_df.columns,
    "Weighting": [f"{w * 100:.2f}%" for w in optimal_weights],
    "Allocated Capital (IDR)": [f"Rp {alloc:,.2f}" for alloc in capital_allocation_idr],
    "Current Price": [f"Rp {current_prices[ticker]:,.2f}" for ticker in combined_df.columns],
    "Shares": [f"{shares[ticker]:.2f}" for ticker in combined_df.columns]
})

st.dataframe(assets_df)

# Portfolio Performance Graphs
st.subheader("📈 Portfolio Performance & Allocation")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Portfolio Cumulative Returns")
    cumulative_returns = (1 + np.dot(log_returns.values, optimal_weights)).cumprod()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cumulative_returns, label="Optimized Portfolio", color="green")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Portfolio Weights Distribution")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(optimal_weights, labels=combined_df.columns, autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title("Optimal Portfolio Weights")
    st.pyplot(fig)

st.subheader("💰 Capital Allocation in IDR")
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(combined_df.columns, capital_allocation_idr, color=plt.cm.Paired.colors)
ax.set_ylabel("Capital Allocation (Rp)")
ax.set_title("Investment Distribution")

# Annotate bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 50000, f"Rp {height:,.2f}", ha="center")

st.pyplot(fig)
