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

# Load custom CSS from external file (styles.css)
with open("styles.css") as f:
    css = f.read()

# Inject the CSS into Streamlit
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Custom CSS for rounded squares for metrics
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f4f7;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-card h3 {
        font-size: 1.2rem;
        font-weight: bold;
        color: #333;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

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

# Initialize adj_close_df as an empty DataFrame
adj_close_df = pd.DataFrame()

# Fetch data for all tickers entered by the user
tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]
for ticker in tickers:
    if ticker:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
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

if adj_close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252  # Annualized covariance matrix

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

    # Capital allocation in IDR
    shares = []
    for i, ticker in enumerate(tickers):
        if '-USD' in ticker:
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

    # Display Metrics in Rounded Squares
    st.subheader('Portfolio Metrics')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Expected Return</h3>
            <p class="value">{portfolio_expected_return:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Risk (Std Dev)</h3>
            <p class="value">{portfolio_risk:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Max Drawdown</h3>
            <p class="value">{max_dd * 100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Sharpe Ratio</h3>
            <p class="value">{sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input):.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <h3>Sortino Ratio</h3>
        <p class="value">{portfolio_sortino:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <h3>Value at Risk (VaR)</h3>
        <p class="value">{portfolio_var * 100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <h3>Expected Shortfall (ES)</h3>
        <p class="value">{portfolio_es * 100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Visualizations
    st.subheader('Portfolio Performance and Allocation')

    # Create 3 columns layout for horizontal stacking
    col1, col2, col3, col4 = st.columns(4)

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
