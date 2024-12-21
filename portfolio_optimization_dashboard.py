import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import requests  # For getting the current exchange rate
import streamlit as st

# Function to get live exchange rate from IDR to USD
def get_exchange_rate():
    api_key = 'ad98bbfc46d9a98d99e9e201'  # Replace with your API key
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/IDR"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        return data['conversion_rates']['USD']
    else:
        raise Exception("Error fetching exchange rate")

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

# Streamlit app structure
st.title('Portfolio Optimization Dashboard')

# User input for tickers
st.subheader("Enter asset tickers")

# Initialize tickers list in session state if not already initialized
if 'tickers' not in st.session_state:
    st.session_state['tickers'] = [""]  # Start with one empty ticker box

# Dynamically add ticker input boxes
for i in range(len(st.session_state['tickers'])):
    ticker = st.text_input(f"Ticker {i+1}", st.session_state['tickers'][i], key=f"ticker_{i}")
    
    # Add a new empty ticker input box if the user presses enter or fills out the current box
    if i == len(st.session_state['tickers']) - 1:
        st.session_state['tickers'].append("")  # Add a new empty ticker box after the last one

# Clean the tickers list (remove empty strings)
tickers = [ticker.strip() for ticker in st.session_state['tickers'] if ticker.strip() != ""]

# User input for risk-free rate
st.subheader("Enter the risk-free rate (in %)")

# Risk-free rate input (convert percentage to decimal)
risk_free_rate_input = st.number_input("Risk-Free Rate (%)", value=6.0, step=0.1) / 100  # Convert percentage to decimal

# User input for investment amount
st.subheader("Enter your investment amount (in IDR)")

investment_amount_idr = st.number_input("Investment Amount (IDR)", value=10000000, step=100000)

# Define the time period for the data
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)

# Download adjusted close price data for each asset
adj_close_df = pd.DataFrame()

# Only fetch data for non-empty tickers
for ticker in tickers:
    if ticker:  # Proceed only if the ticker is non-empty
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if 'Adj Close' exists, otherwise use 'Close'
        if 'Adj Close' in data.columns:
            adj_close_df[ticker] = data['Adj Close']
        elif 'Close' in data.columns:
            adj_close_df[ticker] = data['Close']
        else:
            st.error(f"Data for {ticker} is missing 'Adj Close' and 'Close' columns.")
            continue

# Check if any data was fetched and handle the case if not
if adj_close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    # Calculate log returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    # Covariance matrix for the log returns
    cov_matrix = log_returns.cov() * 252  # Annualize the covariance matrix

    # Fixed portfolio constraints
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

    # Set bounds for the portfolio weights (between 0 and 1 for each asset)
    bounds = [(0.1, 1)] * len(tickers)

    # Optimize portfolio using the negative Sharpe ratio
    initial_weights = np.ones(len(tickers)) / len(tickers)
    optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate_input),
                                 method='SLSQP', constraints=constraints, bounds=bounds)

    # Extract the optimal weights
    optimal_weights = optimized_results.x

    # Get live exchange rate for IDR to USD
    idr_to_usd = get_exchange_rate()

    # Convert the investment amount in IDR to USD
    investment_amount_usd = investment_amount_idr / idr_to_usd

    # Convert capital allocation for each asset to USD
    capital_allocation_usd = optimal_weights * investment_amount_usd

    # Convert the capital allocation back to IDR for displaying
    capital_allocation_idr = capital_allocation_usd * idr_to_usd

    # Display optimal weights and capital allocation
    st.subheader('Optimal Portfolio Weights and Capital Allocation')
    for ticker, weight, capital in zip(tickers, optimal_weights, capital_allocation_idr):
        st.write(f"{ticker}: Weight = {weight:.4f}, Allocated Capital = Rp {capital:,.2f}")

    # Calculate portfolio returns and cumulative returns
    portfolio_returns = np.dot(log_returns.values, optimal_weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Plot the cumulative returns of the portfolio
    st.subheader('Portfolio Cumulative Returns')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_returns, label='Optimized Portfolio')
    ax.set_title('Portfolio Performance (Cumulative Returns)')
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    st.pyplot(fig)

    # Plot the portfolio weights as a pie chart
    st.subheader('Portfolio Weights Distribution')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title('Optimal Portfolio Weights')
    st.pyplot(fig)

    # Plot the capital allocation in IDR as a bar chart
    st.subheader('Capital Allocation in IDR')
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(tickers, capital_allocation_idr, color=plt.cm.Paired.colors)
    ax.set_xlabel('Assets')
    ax.set_ylabel('Allocated Capital (Rp)')
    ax.set_title('Capital Allocation for Investment (in IDR)')

    # Annotate the bars with capital amounts
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 50000, f'Rp {height:,.2f}', 
                 ha='center', va='bottom', fontsize=10, color='black')

    st.pyplot(fig)
