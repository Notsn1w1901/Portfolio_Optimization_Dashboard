import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import streamlit as st
import seaborn as sns

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

# Advanced Metrics Functions
def sortino_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    downside_returns = log_returns[log_returns < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    return (expected_return(weights, log_returns) - risk_free_rate) / downside_deviation

def max_drawdown(cumulative_returns):
    cumulative_returns = pd.Series(cumulative_returns)
    peak = cumulative_returns.cummax()  # Calculate the running maximum of cumulative returns
    drawdown = (cumulative_returns - peak) / peak  # Calculate drawdown
    return drawdown.min()  # Return the maximum drawdown (most negative value)

def value_at_risk(returns, confidence_level=0.95):
    return np.percentile(returns, (1 - confidence_level) * 100)

def expected_shortfall(returns, var):
    return returns[returns <= var].mean()

# Additional Metrics
def beta(portfolio_returns, benchmark_returns):
    # Ensure both returns are pandas Series with aligned indices
    portfolio_returns = pd.Series(portfolio_returns)
    benchmark_returns = pd.Series(benchmark_returns)
    
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns_aligned = portfolio_returns[common_dates]
    benchmark_returns_aligned = benchmark_returns[common_dates]
    
    # Calculate the covariance matrix between the portfolio returns and benchmark returns
    covariance_matrix = np.cov(portfolio_returns_aligned, benchmark_returns_aligned)
    
    # Return the beta as the ratio of the covariance between the portfolio and the benchmark to the variance of the benchmark
    return covariance_matrix[0, 1] / np.var(benchmark_returns_aligned)

def treynor_ratio(weights, log_returns, cov_matrix, risk_free_rate, benchmark_returns):
    portfolio_return = expected_return(weights, log_returns)
    beta_val = beta(np.dot(log_returns.values, weights), benchmark_returns)
    return (portfolio_return - risk_free_rate) / beta_val

def jensen_alpha(weights, log_returns, risk_free_rate, benchmark_returns):
    portfolio_return = expected_return(weights, log_returns)
    beta_val = beta(np.dot(log_returns.values, weights), benchmark_returns)
    market_return = expected_return(np.ones(len(benchmark_returns)) / len(benchmark_returns), benchmark_returns)
    return portfolio_return - (risk_free_rate + beta_val * (market_return - risk_free_rate))

def tracking_error(portfolio_returns, benchmark_returns):
    return np.std(portfolio_returns - benchmark_returns)

def conditional_value_at_risk(returns, var):
    return np.mean(returns[returns <= var])

def skewness(returns):
    return returns.skew()

def kurtosis(returns):
    return returns.kurtosis()

# Streamlit app structure
st.set_page_config(page_title="Portfolio Optimization", layout="wide", initial_sidebar_state="expanded")

# Load custom CSS from external file (styles.css)
with open("styles.css") as f:
    css = f.read()

# Inject the CSS into Streamlit
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Title of the app
st.title('📈 Portfolio Optimization Dashboard')

# Short description of the dashboard functionality
st.markdown("""
    This dashboard allows you to optimize a portfolio of assets by allocating capital across multiple tickers based on historical price data. 
    You can enter asset tickers (e.g., stocks, cryptocurrencies), specify the investment amount in IDR (Indonesian Rupiah), and set the number of years 
    of historical data to be used for analysis. The dashboard will calculate the optimal portfolio weights using the Sharpe ratio optimization method, 
    and display the expected return, risk (standard deviation), and capital allocation for each asset in IDR.
    
    The portfolio is optimized with the objective of maximizing the Sharpe ratio, which represents the best risk-adjusted return. 
    You will also be able to visualize the portfolio's performance, weight distribution, and capital allocation.
    
    Sincerely,  
    **Winston Honadi**
""", unsafe_allow_html=True)

# Sidebar Inputs for User Interactivity
st.sidebar.image("Designer.png", use_container_width=True)  # Add logo to the sidebar
st.sidebar.header("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Enter asset tickers (e.g., BBCA.JK, BTC-USD, TSLA)", "BBCA.JK, BTC-USD")
risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)", value=6.0, step=0.1) / 100  # Convert percentage to decimal
investment_amount_idr = st.sidebar.number_input("Investment Amount (IDR)", value=10000000, step=100000)
years_of_data = st.sidebar.number_input("Years of Data", min_value=1, max_value=20, value=5, step=1)
max_weight = st.sidebar.number_input("Maximum weight per asset (%)", min_value=1, max_value=100, value=50) / 100
min_weight = st.sidebar.number_input("Minimum weight per asset (%)", min_value=0, max_value=100, value=0) / 100
usd_price_idr = st.sidebar.number_input("Current USD Price (IDR)", value=15000, step=100)

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

# Fetch benchmark data (e.g., ^JKSE for Indonesian stock index)
try:
    benchmark_data = yf.download('^JKSE', start=start_date, end=end_date)
    if benchmark_data.empty:
        st.error("No data available for the benchmark (^JKSE). Please check the symbol or your internet connection.")
    else:
        if 'Adj Close' in benchmark_data.columns:
            benchmark_data = benchmark_data['Adj Close']
        else:
            benchmark_data = benchmark_data['Close']
        benchmark_returns = np.log(benchmark_data / benchmark_data.shift(1)).dropna()
except Exception as e:
    st.error(f"Error fetching benchmark data: {e}")

# Check if any data was fetched and handle the case if not
if adj_close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    # Calculate log returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    # Covariance matrix for the log returns
    cov_matrix = log_returns.cov() * 252  # Annualize the covariance matrix

    # Portfolio optimization constraints
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    for i in range(len(tickers)):
        constraints.append({'type': 'ineq', 'fun': lambda weights, i=i: weights[i] - min_weight})
        constraints.append({'type': 'ineq', 'fun': lambda weights, i=i: max_weight - weights[i]})

    # Set bounds for the portfolio weights (between 0 and 1 for each asset)
    bounds = [(min_weight, max_weight)] * len(tickers)

    # Optimize portfolio using the negative Sharpe ratio
    initial_weights = np.ones(len(tickers)) / len(tickers)
    optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate_input),
                                 method='SLSQP', constraints=constraints, bounds=bounds)

    # Extract the optimal weights
    optimal_weights = optimized_results.x

    # Capital allocation for each asset in IDR
    capital_allocation_idr = optimal_weights * investment_amount_idr

    # Calculate the amount of shares based on USD or local assets
    shares = []
    for i, ticker in enumerate(tickers):
        if '-USD' in ticker:
            # For USD-based assets, divide by USD to IDR conversion rate
            shares.append(capital_allocation_idr[i] / usd_price_idr / adj_close_df[ticker].iloc[-1])
        else:
            # For .JK assets, round down to nearest 100 shares
            shares.append(np.floor(capital_allocation_idr[i] / adj_close_df[ticker].iloc[-1] / 100) * 100)

    # Display optimal weights, capital allocation, and amount of shares
    st.subheader('Optimal Portfolio Weights, Capital Allocation, and Shares')
    portfolio_df = pd.DataFrame({
        'Asset': tickers,
        'Weight': optimal_weights,
        'Allocated Capital (IDR)': capital_allocation_idr,
        'Amount of Shares': shares
    })
    st.dataframe(portfolio_df.style.format({'Allocated Capital (IDR)': "Rp {:,.2f}", 'Weight': "{:.4f}", 'Amount of Shares': "{:.8f}"}))

    # Calculate Portfolio Expected Return and Risk
    portfolio_expected_return = expected_return(optimal_weights, log_returns) * 100
    portfolio_risk = standard_deviation(optimal_weights, cov_matrix) * 100

    # Calculate Advanced Metrics
    cumulative_returns = pd.Series((1 + np.dot(log_returns.values, optimal_weights)).cumprod())
    max_dd = max_drawdown(cumulative_returns)
    portfolio_returns = np.dot(log_returns.values, optimal_weights)
    portfolio_var = value_at_risk(portfolio_returns)
    portfolio_es = expected_shortfall(portfolio_returns, portfolio_var)
    portfolio_sortino = sortino_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input)

    # Additional Metrics
    portfolio_beta = beta(portfolio_returns, benchmark_returns)
    portfolio_treynor = treynor_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input, benchmark_returns)
    portfolio_jensen_alpha = jensen_alpha(optimal_weights, log_returns, risk_free_rate_input, benchmark_returns)
    portfolio_tracking_error = tracking_error(portfolio_returns, benchmark_returns)
    portfolio_cvar = conditional_value_at_risk(portfolio_returns, portfolio_var)
    portfolio_skewness = skewness(log_returns)
    portfolio_kurtosis = kurtosis(log_returns)

    # Display Portfolio Metrics
    st.subheader('Portfolio Metrics')
    st.write(f"📊 **Portfolio Expected Return (Annualized)**: {portfolio_expected_return:.2f}%")
    st.write(f"📉 **Portfolio Risk (Standard Deviation)**: {portfolio_risk:.2f}%")
    st.write(f"📊 **Sharpe Ratio**: {sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input):.2f}")
    st.write(f"📈 **Sortino Ratio**: {portfolio_sortino:.2f}")
    st.write(f"⚠️ **Maximum Drawdown**: {max_dd * 100:.2f}%")
    st.write(f"📉 **Value-at-Risk (VaR) at 95% Confidence**: {portfolio_var * 100:.2f}%")
    st.write(f"📉 **Expected Shortfall (ES) at 95% Confidence**: {portfolio_es * 100:.2f}%")
    st.write(f"📈 **Beta**: {portfolio_beta:.2f}")
    st.write(f"📊 **Treynor Ratio**: {portfolio_treynor:.2f}")
    st.write(f"📈 **Jensen's Alpha**: {portfolio_jensen_alpha:.2f}")
    st.write(f"📉 **Tracking Error**: {portfolio_tracking_error:.2f}")
    st.write(f"📊 **Conditional Value-at-Risk (CVaR)**: {portfolio_cvar * 100:.2f}%")
    st.write(f"📉 **Skewness**: {portfolio_skewness:.2f}")
    st.write(f"📈 **Kurtosis**: {portfolio_kurtosis:.2f}")
    
    # Visualization of Portfolio Weights
    st.subheader("Optimal Portfolio Weights (Pie Chart)")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(tickers)))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Performance Plot
    st.subheader('Portfolio Cumulative Returns')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumulative_returns, label='Portfolio', color='blue', linewidth=2)
    ax.set_title('Portfolio Cumulative Returns', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    st.pyplot(fig)
