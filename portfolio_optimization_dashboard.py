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

# Add a checkbox for mutual funds
include_mutual_fund = st.sidebar.checkbox("Include Mutual Fund", value=False)

# Add an input field for the expected return of the mutual fund
mutual_fund_return = 0.0
if include_mutual_fund:
    mutual_fund_return = st.sidebar.number_input("Expected Return of Mutual Fund (%)", value=6.0, step=0.1) / 100

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

# Add mutual fund to the list of tickers if selected
if include_mutual_fund:
    tickers.append("Mutual Fund")

# Fetch data for all tickers
for ticker in tickers:
    try:
        if ticker == "Mutual Fund":
            continue  # Skip fetching data for mutual fund
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Use 'Adj Close' if available, otherwise fallback to 'Close'
        if 'Adj Close' in data.columns:
            price_series = data['Adj Close']
        elif 'Close' in data.columns:
            price_series = data['Close']
        else:
            st.warning(f"No price data available for {ticker}.")
            continue
        
        adj_close_df[ticker] = price_series
        st.write(f"Data fetched for {ticker}")
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")

# Add mutual fund data after fetching all other tickers
if include_mutual_fund:
    # Use the index of adj_close_df to align the mutual fund data
    if not adj_close_df.empty:
        dates = adj_close_df.index
        mutual_fund_values = np.exp(np.log(1 + mutual_fund_return) * np.arange(len(dates)) / 252)
        adj_close_df["Mutual Fund"] = mutual_fund_values
    else:
        st.error("No data available for the selected tickers. Cannot add mutual fund.")
        st.stop()

if adj_close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    # FIX: Add a check to ensure log_returns is not empty before proceeding
    if log_returns.empty:
        st.error("Not enough historical data to perform optimization for the selected timeframe. Please select a longer 'Years of Data' period.")
    else:
        # --- All subsequent code is now safely inside this else block ---
        cov_matrix = log_returns.cov() * 252  # Annualized covariance matrix

        # Set the covariance of the mutual fund to 0 (risk-free)
        if include_mutual_fund:
            if "Mutual Fund" in cov_matrix.index:
                cov_matrix.loc["Mutual Fund", :] = 0
                cov_matrix.loc[:, "Mutual Fund"] = 0

        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        bounds = [(min_weight, max_weight)] * len(tickers)

        initial_weights = np.ones(len(tickers)) / len(tickers)
        optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate_input),
                                     method='SLSQP', constraints=constraints, bounds=bounds)

        optimal_weights = optimized_results.x
        capital_allocation_idr = optimal_weights * investment_amount_idr

        # Capital allocation in IDR
        shares = []
        for i, ticker in enumerate(tickers):
            if ticker == "Mutual Fund":
                shares.append(capital_allocation_idr[i])  # Mutual fund is not traded in shares
            elif '-USD' in ticker:
                shares.append(capital_allocation_idr[i] / usd_price_idr / adj_close_df[ticker].iloc[-1])
            else:
                # Correcting the share calculation for Indonesian stocks (lots of 100)
                shares.append(np.floor(capital_allocation_idr[i] / adj_close_df[ticker].iloc[-1] / 100) * 100)

        portfolio_expected_return = expected_return(optimal_weights, log_returns) * 100
        portfolio_risk = standard_deviation(optimal_weights, cov_matrix) * 100
        cumulative_returns_series = (1 + np.dot(log_returns, optimal_weights)).cumprod()
        max_dd = max_drawdown(cumulative_returns_series)
        portfolio_returns = np.dot(log_returns.values, optimal_weights)
        portfolio_var = value_at_risk(portfolio_returns)
        portfolio_es = expected_shortfall(portfolio_returns, portfolio_var)
        portfolio_sortino = sortino_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input)

        # Create a DataFrame for the asset details
        assets_data = []
        current_prices = {}
        for ticker in tickers:
            if ticker == "Mutual Fund":
                current_prices[ticker] = 1.0  # Unit price of 1 for mutual fund
            else:
                # Use the last available price from the fetched data
                current_prices[ticker] = adj_close_df[ticker].iloc[-1] if ticker in adj_close_df else None

        # Populate the assets data for the table
        for i, ticker in enumerate(tickers):
            weight = optimal_weights[i]
            capital_allocation = capital_allocation_idr[i]
            current_price = current_prices.get(ticker, None)

            if current_price is not None:
                capital_allocation_str = f"Rp {capital_allocation:,.2f}"

                if ticker == "Mutual Fund":
                    shares_value_str = "N/A"
                    currency = "IDR"
                elif '-USD' in ticker:
                    shares_value = capital_allocation / usd_price_idr / current_price
                    currency = 'USD'
                    shares_value_str = f"{shares_value:.8f}"
                else:
                    shares_value = np.floor(capital_allocation / current_price / 100) * 100 if current_price > 0 else 0
                    currency = 'IDR'
                    shares_value_str = f"{int(shares_value)}"

                assets_data.append([ticker, f"{weight * 100:.2f}%", capital_allocation_str, f"{current_price:,.2f} {currency}", shares_value_str])
            else:
                assets_data.append([ticker, f"{weight * 100:.2f}%", "N/A", "N/A", "N/A"])

        assets_df = pd.DataFrame(assets_data, columns=["Asset", "Weighting", "Allocated Capital", "Current Price", "Shares"])

        # Display the table
        st.subheader('📝 Asset Details')
        st.dataframe(assets_df)

        st.subheader('📊 Portfolio Metrics')
        
        # First row (2 columns, equal size)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card metric-return">
                <div class="icon">📈</div>
                <h3>Expected Return</h3>
                <p class="value">{portfolio_expected_return:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
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
        with col3:
            st.markdown(f"""
            <div class="metric-card metric-drawdown">
                <div class="icon">⛔</div>
                <h3>Max Drawdown</h3>
                <p class="value">{max_dd * 100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card metric-sharpe">
                <div class="icon">📊</div>
                <h3>Sharpe Ratio</h3>
                <p class="value">{sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input):.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        # Third row (3 columns, equal size)
        col5, col6, col7 = st.columns(3)
        with col5:
            st.markdown(f"""
            <div class="metric-card metric-other">
                <div class="icon">⚡</div>
                <h3>Sortino Ratio</h3>
                <p class="value">{portfolio_sortino:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col6:
            st.markdown(f"""
            <div class="metric-card metric-other">
                <div class="icon">💥</div>
                <h3>Value at Risk (VaR)</h3>
                <p class="value">{portfolio_var * 100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col7:
            st.markdown(f"""
            <div class="metric-card metric-other">
                <div class="icon">💸</div>
                <h3>Expected Shortfall (ES)</h3>
                <p class="value">{portfolio_es * 100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader('Portfolio Performance and Allocation')
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader('Portfolio Cumulative Returns')
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(cumulative_returns_series.index, cumulative_returns_series.values, label='Optimized Portfolio', color="#4CAF50", linewidth=2)
            ax.set_title('Portfolio Performance (Cumulative Returns)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Cumulative Returns', fontsize=12)
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            st.subheader('Portfolio Weights Distribution')
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.set_title('Optimal Portfolio Weights', fontsize=14, fontweight='bold')
            st.pyplot(fig)

        with col3:
            st.subheader('Capital Allocation in IDR')
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(tickers, capital_allocation_idr, color=plt.cm.Paired.colors)
            ax.set_xlabel('Assets', fontsize=12)
            ax.set_ylabel('Allocated Capital (Rp)', fontsize=14)
            ax.set_title('Capital Allocation for Investment (in IDR)', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Annotate the bars with capital amounts
            for bar in bars:
                height = bar.get_height()
                # THIS IS THE CORRECTED INDENTATION
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'Rp {height:,.0f}',
                        ha='center', va='bottom', fontsize=9, color='black')
            
            st.pyplot(fig)
