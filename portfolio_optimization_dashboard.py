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

# Short description of the dashboard functionality
st.write("""
This dashboard allows you to optimize a portfolio of assets by allocating capital across multiple tickers based on historical price data. 
You can enter asset tickers (e.g., stocks, cryptocurrencies), specify the investment amount in IDR (Indonesian Rupiah), and set the number of years 
of historical data to be used for analysis. The dashboard will calculate the optimal portfolio weights using the Sharpe ratio optimization method, 
and display the expected return, risk (standard deviation), and capital allocation for each asset in both IDR and USD.

The portfolio is optimized with the objective of maximizing the Sharpe ratio, which represents the best risk-adjusted return. 
You will also be able to visualize the portfolio's performance, weight distribution, and capital allocation.

Sincerely,

Winston Honadi
""")

# User input for tickers
st.subheader("Enter asset tickers (separated by commas)")

# Input for multiple tickers in one text box
tickers_input = st.text_input("Tickers (e.g., BBCA.JK, BTC-USD, TSLA)", "BBCA.JK, BTC-USD")
tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]

# User input for risk-free rate
st.subheader("Enter the risk-free rate (in %)")

# Risk-free rate input (convert percentage to decimal)
risk_free_rate_input = st.number_input("Risk-Free Rate (%)", value=6.0, step=0.1) / 100  # Convert percentage to decimal

# User input for investment amount
st.subheader("Enter your investment amount (in IDR)")

investment_amount_idr = st.number_input("Investment Amount (IDR)", value=10000000, step=100000)

# User input for the number of years of data to be used
st.subheader("Enter the number of years of data to use")

years_of_data = st.number_input("Years of Data", min_value=1, max_value=20, value=5, step=1)  # User-defined years

# Define the time period for the data
end_date = datetime.today()
start_date = end_date - timedelta(days=years_of_data * 365)  # Use user input for years

# Download adjusted close price data for each asset
adj_close_df = pd.DataFrame()

# Fetch data for all tickers entered by the user
for ticker in tickers:
    if ticker:  # Proceed only if the ticker is non-empty
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Check if 'Adj Close' exists, otherwise use 'Close'
            if 'Adj Close' in data.columns:
                adj_close_df[ticker] = data['Adj Close']
            elif 'Close' in data.columns:
                adj_close_df[ticker] = data['Close']
            else:
                st.warning(f"Data for {ticker} is missing 'Adj Close' and 'Close' columns.")
                continue
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            continue

# Check if any data was fetched and handle the case if not
if adj_close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    # Calculate log returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    # Covariance matrix for the log returns
    cov_matrix = log_returns.cov() * 252  # Annualize the covariance matrix

    # Get live exchange rate for IDR to USD
    idr_to_usd = get_exchange_rate()

    # Convert the investment amount in IDR to USD
    investment_amount_usd = investment_amount_idr / idr_to_usd

    # Define a function to calculate portfolio return and risk
    def portfolio_metrics(weights, log_returns, cov_matrix):
        portfolio_return = expected_return(weights, log_returns)
        portfolio_risk = standard_deviation(weights, cov_matrix)
        return portfolio_return, portfolio_risk

    # Generate the Efficient Frontier with error handling
    target_risks = np.linspace(0.01, 0.50, 100)  # 100 risk levels from 1% to 50%
    efficient_returns = []
    efficient_risks = []

    for target_risk in target_risks:
        # Minimize the negative expected return for each target risk (Markowitz optimization)
        def objective(weights):
            port_return, port_risk = portfolio_metrics(weights, log_returns, cov_matrix)
            # If the portfolio risk is greater than the target risk, return a high penalty (inf)
            return -port_return if port_risk <= target_risk else np.inf

        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1)] * len(tickers)  # Weights between 0 and 1
        initial_weights = np.ones(len(tickers)) / len(tickers)  # Starting guess: equally weighted

        # Optimization with error handling for each target risk
        try:
            result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                efficient_return, efficient_risk = portfolio_metrics(result.x, log_returns, cov_matrix)
                efficient_returns.append(efficient_return)
                efficient_risks.append(efficient_risk)
            else:
                st.warning(f"Optimization failed for target risk level {target_risk:.2f}.")
                efficient_returns.append(np.nan)  # Append NaN if optimization fails
                efficient_risks.append(np.nan)

        except Exception as e:
            st.warning(f"Error in optimization for target risk level {target_risk:.2f}: {e}")
            efficient_returns.append(np.nan)
            efficient_risks.append(np.nan)

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

    # Convert capital allocation for each asset to USD
    capital_allocation_usd = optimal_weights * investment_amount_usd

    # Convert the capital allocation back to IDR for displaying
    capital_allocation_idr = capital_allocation_usd * idr_to_usd

    # Display optimal weights and capital allocation
    st.subheader('Optimal Portfolio Weights and Capital Allocation')
    for ticker, weight, capital in zip(tickers, optimal_weights, capital_allocation_idr):
        st.write(f"{ticker}: Weight = {weight:.4f}, Allocated Capital = Rp {capital:,.2f}")

    # Calculate Portfolio Expected Return and Risk
    portfolio_expected_return = expected_return(optimal_weights, log_returns)
    portfolio_risk = standard_deviation(optimal_weights, cov_matrix)

    # Display Portfolio Expected Return and Risk
    st.subheader('Portfolio Metrics')
    st.write(f"Portfolio Expected Return (Annualized): {portfolio_expected_return:.2f}%")
    st.write(f"Portfolio Risk (Standard Deviation): {portfolio_risk:.2f}")

    # Calculate portfolio returns and cumulative returns
    portfolio_returns = np.dot(log_returns.values, optimal_weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Plot the Portfolio Cumulative Returns
    st.subheader('Portfolio Cumulative Returns')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_returns, label='Optimized Portfolio')
    ax.set_title('Portfolio Performance (Cumulative Returns)')
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    st.pyplot(fig)

    # Plot the Markowitz Efficient Frontier graph below the cumulative returns graph
    st.subheader('Markowitz Efficient Frontier')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(efficient_risks, efficient_returns, label="Efficient Frontier", color='green')
    ax.set_xlabel('Risk (Standard Deviation)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Markowitz Efficient Frontier')
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
