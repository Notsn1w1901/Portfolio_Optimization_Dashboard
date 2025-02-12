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

def sortino_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    downside_returns = log_returns[log_returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    return (expected_return(weights, log_returns) - risk_free_rate) / downside_deviation

def max_drawdown(cumulative_returns):
    cumulative_returns = pd.Series(cumulative_returns)
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def value_at_risk(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

def expected_shortfall(returns, var):
    return returns[returns <= var].mean()

# Streamlit app structure
st.set_page_config(page_title="Portfolio Optimization", layout="wide", initial_sidebar_state="expanded")

st.title('📈 Portfolio Optimization Dashboard')

st.sidebar.header("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Enter asset tickers (e.g., BBCA.JK, BTC-USD, TSLA)", "BBCA.JK, BTC-USD")
risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)", value=6.0, step=0.1) / 100
investment_amount_idr = st.sidebar.number_input("Investment Amount (IDR)", value=10000000, step=100000)
years_of_data = st.sidebar.number_input("Years of Data", min_value=1, max_value=20, value=5, step=1)
max_weight = st.sidebar.number_input("Maximum weight per asset (%)", min_value=1, max_value=100, value=50) / 100
min_weight = st.sidebar.number_input("Minimum weight per asset (%)", min_value=0, max_value=100, value=0) / 100
usd_price_idr = st.sidebar.number_input("Current USD Price (IDR)", value=15000, step=100)

# Mutual Fund Option
include_mutual_fund = st.sidebar.checkbox("Include Mutual Fund (5% Return)")

end_date = datetime.today()
start_date = end_date - timedelta(days=years_of_data * 365)

adj_close_df = pd.DataFrame()
tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]

# Fetch stock data
for ticker in tickers:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if 'Adj Close' in data.columns:
            adj_close_df[ticker] = data['Adj Close']
        elif 'Close' in data.columns:
            adj_close_df[ticker] = data['Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")

if adj_close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252

    if include_mutual_fund:
        tickers.append("Mutual Fund")
        mutual_fund_return = 0.05
        log_returns["Mutual Fund"] = mutual_fund_return / 252
        zero_volatility = np.zeros((len(tickers), len(tickers)))
        zero_volatility[:-1, :-1] = cov_matrix
        cov_matrix = zero_volatility

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

    shares = []
    for i, ticker in enumerate(tickers):
        if ticker == "Mutual Fund":
            shares.append(capital_allocation_idr[i] / investment_amount_idr)  # % of capital
        elif '-USD' in ticker:
            shares.append(capital_allocation_idr[i] / usd_price_idr / adj_close_df[ticker].iloc[-1])
        else:
            shares.append(np.floor(capital_allocation_idr[i] / adj_close_df[ticker].iloc[-1] / 100) * 100)

    portfolio_expected_return = expected_return(optimal_weights, log_returns) * 100
    portfolio_risk = standard_deviation(optimal_weights, cov_matrix) * 100
    cumulative_returns = pd.Series((1 + np.dot(log_returns.values, optimal_weights)).cumprod())
    max_dd = max_drawdown(cumulative_returns)
    portfolio_returns = np.dot(log_returns.values, optimal_weights)
    portfolio_var = value_at_risk(portfolio_returns)
    portfolio_es = expected_shortfall(portfolio_returns, portfolio_var)
    portfolio_sortino = sortino_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input)

    # Display portfolio metrics
    st.subheader('📊 Portfolio Metrics')
    st.write(f"**Expected Return:** {portfolio_expected_return:.2f}%")
    st.write(f"**Risk (Standard Deviation):** {portfolio_risk:.2f}%")
    st.write(f"**Max Drawdown:** {max_dd * 100:.2f}%")
    st.write(f"**Sharpe Ratio:** {sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input):.2f}")
    st.write(f"**Sortino Ratio:** {portfolio_sortino:.2f}")
    st.write(f"**Value at Risk (VaR):** {portfolio_var * 100:.2f}%")
    st.write(f"**Expected Shortfall (ES):** {portfolio_es * 100:.2f}%")

    # Capital allocation pie chart
    st.subheader('Portfolio Weights Distribution')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title('Optimal Portfolio Weights')
    st.pyplot(fig)

    # Capital allocation bar chart
    st.subheader('Capital Allocation in IDR')
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(tickers, capital_allocation_idr, color=plt.cm.Paired.colors)
    ax.set_xlabel('Assets')
    ax.set_ylabel('Allocated Capital (Rp)')
    ax.set_title('Capital Allocation for Investment (in IDR)')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50000, f'Rp {bar.get_height():,.2f}', 
                ha='center', va='bottom', fontsize=10, color='black')
    st.pyplot(fig)
