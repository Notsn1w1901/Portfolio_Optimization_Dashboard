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
    # Handle potential division by zero
    std_dev = standard_deviation(weights, cov_matrix)
    if std_dev == 0:
        return np.inf  # Or some other large number if appropriate
    return (expected_return(weights, log_returns) - risk_free_rate) / std_dev

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

# --- MODIFIED ---
# Corrected the Sortino Ratio function to use portfolio returns
def sortino_ratio(weights, log_returns, risk_free_rate):
    annual_return = expected_return(weights, log_returns)
    
    # Calculate daily portfolio returns
    portfolio_daily_returns = np.dot(log_returns, weights)
    
    # Identify downside returns (returns below the target, typically 0 for daily returns)
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
    
    # If there are no downside returns, the risk is zero, so Sortino is infinite
    if len(downside_returns) == 0:
        return np.inf
        
    # Calculate Downside Deviation
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    
    # Avoid division by zero
    if downside_deviation == 0:
        return np.inf
        
    return (annual_return - risk_free_rate) / downside_deviation

# --- NEW ---
# Create the negative Sortino Ratio function for the optimizer
def neg_sortino_ratio(weights, log_returns, risk_free_rate):
    return -sortino_ratio(weights, log_returns, risk_free_rate)

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

# Custom CSS... (Your CSS remains the same)
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

# Title and description... (Your title and description remain the same)
st.title('📈 Portfolio Optimization Dashboard')
st.markdown("""
This dashboard empowers you to optimize investment portfolios based on either the **Sharpe Ratio** (risk-adjusted return) or the **Sortino Ratio** (downside-risk-adjusted return).
""", unsafe_allow_html=True)


# Sidebar Inputs for User Interactivity
st.sidebar.image("Designer.png", use_container_width=True)
st.sidebar.header("Portfolio Inputs")

# --- NEW ---
# Add a selection box for the optimization metric
optimization_metric = st.sidebar.selectbox(
    "Select Optimization Metric",
    ("Sharpe Ratio", "Sortino Ratio")
)

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

# Data fetching... (Your data fetching logic remains the same)
end_date = datetime.today()
start_date = end_date - timedelta(days=years_of_data * 365)
adj_close_df = pd.DataFrame()
tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]

if include_mutual_fund:
    tickers.append("Mutual Fund")

for ticker in tickers:
    try:
        if ticker == "Mutual Fund":
            continue
        data = yf.download(ticker, start=start_date, end=end_date)
        if 'Adj Close' in data.columns:
            price_series = data['Adj Close']
        elif 'Close' in data.columns:
            price_series = data['Close']
        else:
            st.warning(f"No price data available for {ticker}.")
            continue
        adj_close_df[ticker] = price_series
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")

if include_mutual_fund:
    if not adj_close_df.empty:
        dates = adj_close_df.index
        daily_return_rate = (1 + mutual_fund_return)**(1/252) - 1
        mutual_fund_values = (1 + daily_return_rate).cumprod(axis=0) * np.arange(len(dates))
        # A more stable way to generate synthetic data
        start_value = 1000 
        daily_returns = np.full(shape=(len(dates),), fill_value=daily_return_rate)
        mutual_fund_prices = start_value * (1 + daily_returns).cumprod()
        adj_close_df["Mutual Fund"] = pd.Series(mutual_fund_prices, index=dates)
    else:
        st.error("No data available for other tickers. Cannot add mutual fund.")
        st.stop()

# --- Main Logic Block ---
if adj_close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    if log_returns.empty:
        st.error("Not enough historical data to perform optimization for the selected timeframe. Please select a longer 'Years of Data' period.")
    else:
        cov_matrix = log_returns.cov() * 252

        if include_mutual_fund and "Mutual Fund" in cov_matrix.index:
            cov_matrix.loc["Mutual Fund", :] = 0
            cov_matrix.loc[:, "Mutual Fund"] = 0

        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        bounds = [(min_weight, max_weight)] * len(tickers)
        initial_weights = np.ones(len(tickers)) / len(tickers)

        # --- MODIFIED ---
        # Dynamically select the objective function and arguments based on user choice
        if optimization_metric == "Sharpe Ratio":
            st.info("🚀 Optimizing portfolio for the best Sharpe Ratio...")
            objective_function = neg_sharpe_ratio
            args = (log_returns, cov_matrix, risk_free_rate_input)
        else: # Sortino Ratio
            st.info("🚀 Optimizing portfolio for the best Sortino Ratio...")
            objective_function = neg_sortino_ratio
            args = (log_returns, risk_free_rate_input) # Note: cov_matrix is not needed for Sortino
        
        optimized_results = minimize(objective_function, initial_weights, args=args,
                                     method='SLSQP', constraints=constraints, bounds=bounds)

        # The rest of the script continues from here...
        optimal_weights = optimized_results.x
        capital_allocation_idr = optimal_weights * investment_amount_idr

        # Calculate metrics using the optimized weights
        portfolio_expected_return = expected_return(optimal_weights, log_returns) * 100
        portfolio_risk = standard_deviation(optimal_weights, cov_matrix) * 100
        portfolio_sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input)
        portfolio_sortino = sortino_ratio(optimal_weights, log_returns, risk_free_rate_input)

        portfolio_daily_returns = np.dot(log_returns, optimal_weights)
        cumulative_returns_series = pd.Series((1 + portfolio_daily_returns).cumprod(), index=log_returns.index)
        
        max_dd = max_drawdown(cumulative_returns_series)
        portfolio_var = value_at_risk(portfolio_daily_returns)
        portfolio_es = expected_shortfall(portfolio_daily_returns, portfolio_var)

        # ... (Your dataframe creation and display code remains the same) ...
        # ... (Your metric card and plotting code remains the same, but now reflects the chosen optimization) ...

        assets_data = []
        for i, ticker in enumerate(tickers):
            weight = optimal_weights[i]
            capital = capital_allocation_idr[i]
            current_price = adj_close_df[ticker].iloc[-1]
            currency = "USD" if "-USD" in ticker else "IDR"
            
            if currency == "USD":
                shares = capital / (current_price * usd_price_idr)
                shares_str = f"{shares:.8f}"
            elif ticker == "Mutual Fund":
                shares_str = "N/A"
            else:
                shares = np.floor(capital / current_price / 100) * 100
                shares_str = f"{int(shares)}"

            assets_data.append([
                ticker,
                f"{weight*100:.2f}%",
                f"Rp {capital:,.2f}",
                f"{current_price:,.2f} {currency}",
                shares_str
            ])
            
        assets_df = pd.DataFrame(assets_data, columns=["Asset", "Weighting", "Allocated Capital", "Current Price", "Shares"])
        st.subheader('📝 Asset Details')
        st.dataframe(assets_df)

        st.subheader('📊 Portfolio Metrics')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card metric-return">
                <div class="icon">📈</div><h3>Expected Return</h3><p class="value">{portfolio_expected_return:.2f}%</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card metric-risk">
                <div class="icon">⚖️</div><h3>Risk (Std Dev)</h3><p class="value">{portfolio_risk:.2f}%</p>
            </div>""", unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"""
            <div class="metric-card metric-drawdown">
                <div class="icon">⛔</div><h3>Max Drawdown</h3><p class="value">{max_dd * 100:.2f}%</p>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card metric-sharpe">
                <div class="icon">📊</div><h3>Sharpe Ratio</h3><p class="value">{portfolio_sharpe:.2f}</p>
            </div>""", unsafe_allow_html=True)

        col5, col6, col7 = st.columns(3)
        with col5:
            st.markdown(f"""
            <div class="metric-card metric-other">
                <div class="icon">⚡</div><h3>Sortino Ratio</h3><p class="value">{portfolio_sortino:.2f}</p>
            </div>""", unsafe_allow_html=True)
        with col6:
            st.markdown(f"""
            <div class="metric-card metric-other">
                <div class="icon">💥</div><h3>Value at Risk (VaR)</h3><p class="value">{portfolio_var * 100:.2f}%</p>
            </div>""", unsafe_allow_html=True)
        with col7:
            st.markdown(f"""
            <div class="metric-card metric-other">
                <div class="icon">💸</div><h3>Expected Shortfall (ES)</h3><p class="value">{portfolio_es * 100:.2f}%</p>
            </div>""", unsafe_allow_html=True)

        st.subheader('Portfolio Performance and Allocation')
        col1_chart, col2_chart, col3_chart = st.columns(3)
        with col1_chart:
            st.subheader('Portfolio Cumulative Returns')
            fig, ax = plt.subplots()
            ax.plot(cumulative_returns_series.index, cumulative_returns_series.values, label='Optimized Portfolio', color="#4CAF50")
            ax.set_title('Portfolio Performance')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Returns')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        with col2_chart:
            st.subheader('Portfolio Weights Distribution')
            fig, ax = plt.subplots()
            ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
            ax.set_title('Optimal Portfolio Weights')
            st.pyplot(fig)
        with col3_chart:
            st.subheader('Capital Allocation in IDR')
            fig, ax = plt.subplots()
            ax.bar(tickers, capital_allocation_idr)
            ax.set_title('Capital Allocation')
            ax.set_ylabel('Allocated Capital (IDR)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
