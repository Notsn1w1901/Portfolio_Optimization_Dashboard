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

# Advanced Metrics Functions
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

# Custom CSS for rounded squares for metrics
st.markdown("""
<style>
    /* Metric Card Styles */
    .metric-card {
        background: linear-gradient(145deg, #6a8dff, #4CAF50);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }

    .metric-card h3 {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
    }

    .metric-card .icon {
        font-size: 2rem;
        color: white;
        margin-bottom: 10px;
    }

    /* Green for return, Red for risk */
    .metric-return {
        background: linear-gradient(145deg, #81C784, #388E3C);
    }

    .metric-risk {
        background: linear-gradient(145deg, #FF7043, #D32F2F);
    }

    .metric-drawdown {
        background: linear-gradient(145deg, #FFB74D, #F57C00);
    }

    .metric-sharpe {
        background: linear-gradient(145deg, #64B5F6, #1976D2);
    }

    .metric-other {
        background: linear-gradient(145deg, #80DEEA, #26C6DA);
    }
</style>
""", unsafe_allow_html=True)

# Title of the app
st.title('📈 Portfolio Optimization Dashboard')

# Short description of the dashboard functionality
st.markdown("""
This dashboard empowers you to optimize investment portfolios.

Key Features:

    • Allocate capital across multiple assets (stocks, cryptocurrencies, etc.)

    • Define investment amount in IDR.

    • Specify historical price data timeframe.

    • Employ Sharpe Ratio optimization for optimal asset allocation.


Outputs:

    • Expected portfolio return.

    • Portfolio risk (standard deviation).

    • Capital allocation per asset in IDR.

    • Visualizations of portfolio performance, weight distribution, and capital allocation.


Objective: Maximize risk-adjusted returns by optimizing the Sharpe Ratio.
""", unsafe_allow_html=True)

# Sidebar Inputs for User Interactivity
st.sidebar.image("Designer.png", use_container_width=True)
st.sidebar.header("Portfolio Inputs")
tickers_input = st.sidebar.text_input("Enter asset tickers (e.g., BBCA.JK, BTC-USD, TSLA)", "BBCA.JK, BTC-USD")
risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)", value=6.0, step=0.1) / 100
investment_amount_idr = st.sidebar.number_input("Investment Amount (IDR)", value=10000000, step=100000)
years_of_data = st.sidebar.number_input("Years of Data", min_value=1, max_value=20, value=5, step=1)
max_weight = st.sidebar.number_input("Maximum weight per asset (%)", min_value=1, max_value=100, value=50) / 100
min_weight = st.sidebar.number_input("Minimum weight per asset (%)", min_value=0, max_value=100, value=0) / 100
usd_price_idr = st.sidebar.number_input("Current USD Price (IDR)", value=15000, step=100)

# Define the time period for the data
end_date = datetime.today()
start_date = end_date - timedelta(days=years_of_data * 365)

# Fetch adjusted close and close prices
adj_close_df = pd.DataFrame()
close_df = pd.DataFrame()

# Fetch data for all tickers entered by the user
tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]
for ticker in tickers:
    if ticker:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            print(f"Data for {ticker}:\n{data.head()}")
            # Continue with processing as before...
            # Use 'Adj Close' for portfolio optimization and 'Close' for share calculation
            if 'Adj Close' in data.columns:
                adj_close_df[ticker] = data['Adj Close']
            else:
                st.warning(f"Adjusted Close data is missing for {ticker}.")
                continue

            if 'Close' in data.columns:
                close_df[ticker] = data['Close']
            else:
                st.warning(f"Close data is missing for {ticker}.")
                continue

            # Handle NaN values by forward and backward filling
            if adj_close_df[ticker].isnull().any():
                st.warning(f"NaN values found in data for {ticker}. Filling NaNs with the previous value.")
                adj_close_df[ticker].fillna(method='ffill', inplace=True)  # Forward fill to handle NaN
                adj_close_df[ticker].fillna(method='bfill', inplace=True)  # Backward fill if necessary
            if close_df[ticker].isnull().any():
                st.warning(f"NaN values found in data for {ticker}. Filling NaNs with the previous value.")
                close_df[ticker].fillna(method='ffill', inplace=True)  # Forward fill to handle NaN
                close_df[ticker].fillna(method='bfill', inplace=True)  # Backward fill if necessary

        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            continue

if adj_close_df.empty or close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()  # Log returns for optimization
    cov_matrix = log_returns.cov() * 252  # Annualized covariance matrix for optimization

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

    # Calculate shares using close price (from the previous day)
    shares = []
    for i, ticker in enumerate(tickers):
        capital_allocation = capital_allocation_idr[i]
        
        # Get the last 'Close' price for each asset
        close_price = close_df[ticker].iloc[-1]  # Close price on the last available day
        
        # Adjust the share calculation for USD-based assets (cryptos or USD stocks)
        if '-USD' in ticker:
            close_price = close_price * usd_price_idr  # Convert the price to IDR by multiplying with the USD to IDR rate
        
        # Calculate the number of shares to purchase based on the capital allocation
        if '-USD' in ticker:  # For cryptocurrencies or USD-based assets
            shares.append(capital_allocation / close_price)  # Allocated capital divided by the price in IDR
        else:  # For stocks
            shares.append(np.floor(capital_allocation / close_price / 100) * 100)  # Round down to nearest 100 shares

    portfolio_expected_return = expected_return(optimal_weights, log_returns) * 100
    portfolio_risk = standard_deviation(optimal_weights, cov_matrix) * 100
    cumulative_returns = pd.Series((1 + np.dot(log_returns.values, optimal_weights)).cumprod())
    max_dd = max_drawdown(cumulative_returns)
    portfolio_returns = np.dot(log_returns.values, optimal_weights)
    portfolio_var = value_at_risk(portfolio_returns)
    portfolio_es = expected_shortfall(portfolio_returns, portfolio_var)
    portfolio_sortino = sortino_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input)

    # Prepare Asset Data for display
    assets_data = []
    for i, ticker in enumerate(tickers):
        weight = optimal_weights[i]
        capital_allocation = capital_allocation_idr[i]
        
        # For cryptocurrencies or USD-based assets, use the adjusted price in IDR
        share_price = close_df[ticker].iloc[-1]  # Using Close price for share calculation
        
        if '-USD' in ticker:
            share_price = share_price * usd_price_idr  # Convert the price to IDR by multiplying with the USD to IDR rate
        
        assets_data.append([ticker, f"{weight * 100:.2f}%", f"Rp {capital_allocation:,.2f}", f"Rp {share_price:,.2f}", shares[i]])

    # Convert the asset data to a pandas DataFrame
    assets_df = pd.DataFrame(assets_data, columns=["Asset", "Weighting", "Allocated Capital (IDR)", "Price (IDR)", "Shares"])

    # Display the table with asset details
    st.subheader('📝 Asset Details')
    st.dataframe(assets_df)

    st.subheader('📊 Portfolio Metrics')

    # First row (2 columns, equal size)
    col1, col2 = st.columns(2)
    
    # Expected Return
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-return">
            <div class="icon">📈</div>
            <h3>Expected Return</h3>
            <p class="value">{portfolio_expected_return:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk (Standard Deviation)
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-risk">
            <div class="icon">⚖️</div>
            <h3>Risk (Std Dev)</h3>
            <p class="value">{portfolio_risk:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Second row (2 columns, equal size)
    col3, col4 = st.columns(2)
    
    # Max Drawdown
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-drawdown">
            <div class="icon">⛔</div>
            <h3>Max Drawdown</h3>
            <p class="value">{max_dd:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sharpe Ratio
    with col4:
        st.markdown(f"""
        <div class="metric-card metric-sharpe">
            <div class="icon">📊</div>
            <h3>Sharpe Ratio</h3>
            <p class="value">{sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input):.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional metrics row
    col5, col6 = st.columns(2)
    
    # Value at Risk
    with col5:
        st.markdown(f"""
        <div class="metric-card metric-other">
            <div class="icon">⚠️</div>
            <h3>Value at Risk (95%)</h3>
            <p class="value">Rp {portfolio_var:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Expected Shortfall
    with col6:
        st.markdown(f"""
        <div class="metric-card metric-other">
            <div class="icon">💥</div>
            <h3>Expected Shortfall (95%)</h3>
            <p class="value">Rp {portfolio_es:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader('📊 Portfolio Performance')

    # Create 3 columns layout for horizontal stacking
    col1, col2, col3 = st.columns(3)

    with col1:
        # Cumulative Returns Graph
        st.subheader('Portfolio Cumulative Returns')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(cumulative_returns, label='Optimized Portfolio', color="#4CAF50", linewidth=2)
        ax.set_title('Portfolio Performance (Cumulative Returns)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (Days)', fontsize=12)
        ax.set_ylabel('Cumulative Returns', fontsize=12)
        ax.legend()
        st.pyplot(fig)

    with col2:
        # Portfolio Weights Distribution Graph (Pie chart)
        st.subheader('Portfolio Weights Distribution')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
        ax.set_title('Optimal Portfolio Weights', fontsize=14, fontweight='bold')
        st.pyplot(fig)

    with col3:
        # Capital Allocation in IDR Graph (Bar chart)
        st.subheader('Capital Allocation in IDR')
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(tickers, capital_allocation_idr, color=plt.cm.Paired.colors)
        ax.set_xlabel('Assets', fontsize=12)
        ax.set_ylabel('Allocated Capital (Rp)', fontsize=14)
        ax.set_title('Capital Allocation for Investment (in IDR)', fontsize=14, fontweight='bold')

        # Annotate the bars with capital amounts
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 50000, f'Rp {height:,.2f}', 
                     ha='center', va='bottom', fontsize=10, color='black')

        st.pyplot(fig)
