import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import streamlit as st
from streamlit import cache_data

# --- NEW ROBUST PRICE FUNCTION ---
@cache_data # Cache the results to avoid re-fetching on every interaction
def get_current_price(ticker):
    """
    Fetches the most recent price for a given ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        # Use history() for the most reliable recent price
        todays_data = stock.history(period='1d')
        if not todays_data.empty:
            return todays_data['Close'].iloc[-1]
        
        # Fallback to info dictionary if history is empty
        info = stock.info
        if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return info['regularMarketPrice']
        elif 'previousClose' in info and info['previousClose'] is not None:
            return info['previousClose']
        else:
            return None # No price found
            
    except Exception as e:
        return None

# --- PORTFOLIO STATISTICS FUNCTIONS ---
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    std_dev = standard_deviation(weights, cov_matrix)
    if std_dev == 0:
        return np.inf
    return (expected_return(weights, log_returns) - risk_free_rate) / std_dev

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

def sortino_ratio(weights, log_returns, risk_free_rate):
    annual_return = expected_return(weights, log_returns)
    portfolio_daily_returns = np.dot(log_returns, weights)
    downside_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
        
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
    
    if downside_deviation == 0:
        return np.inf
        
    return (annual_return - risk_free_rate) / downside_deviation

def neg_sortino_ratio(weights, log_returns, risk_free_rate):
    return -sortino_ratio(weights, log_returns, risk_free_rate)

def max_drawdown(cumulative_returns):
    cumulative_returns = pd.Series(cumulative_returns)
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def value_at_risk(returns, confidence_level=0.95):
    if len(returns) == 0:
        return 0
    return np.percentile(returns, (1 - confidence_level) * 100)

def expected_shortfall(returns, var):
    if len(returns) == 0:
        return 0
    return returns[returns <= var].mean()

# --- STREAMLIT APP UI ---
st.set_page_config(page_title="Portfolio Optimization", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Metric Card Styles */
    .metric-card {
        background: linear-gradient(145deg, #2e335b, #4a5080);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s ease-in-out;
        color: white;
    }
    .metric-card:hover { transform: scale(1.05); }
    .metric-card h3 { font-size: 1.5rem; font-weight: bold; color: white; margin-bottom: 10px; }
    .metric-card .value { font-size: 2rem; font-weight: bold; color: #ffffff; }
    .metric-card .icon { font-size: 2rem; color: white; margin-bottom: 10px; }
    .metric-return { background: linear-gradient(145deg, #4CAF50, #2E7D32); }
    .metric-risk { background: linear-gradient(145deg, #f44336, #c62828); }
    .metric-drawdown { background: linear-gradient(145deg, #FF9800, #EF6C00); }
    .metric-sharpe { background: linear-gradient(145deg, #2196F3, #1565C0); }
    .metric-other { background: linear-gradient(145deg, #00BCD4, #00838F); }
</style>
""", unsafe_allow_html=True)

st.title('📈 Portfolio Optimization Dashboard')
st.markdown("This dashboard empowers you to optimize investment portfolios based on either the **Sharpe Ratio** (risk-adjusted return) or the **Sortino Ratio** (downside-risk-adjusted return).")

# --- SIDEBAR INPUTS ---
st.sidebar.image("Designer.png", use_container_width=True)
st.sidebar.header("Portfolio Inputs")

optimization_metric = st.sidebar.selectbox("Select Optimization Metric", ("Sharpe Ratio", "Sortino Ratio"))
include_mutual_fund = st.sidebar.checkbox("Include Mutual Fund", value=False)

if include_mutual_fund:
    mutual_fund_return = st.sidebar.number_input("Expected Annual Return of Mutual Fund (%)", value=8.0, step=0.1) / 100

tickers_input = st.sidebar.text_input("Enter asset tickers (e.g., BBCA.JK, BTC-USD)", "BBCA.JK,BTC-USD,ITMG.JK")
risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)", value=6.0, step=0.1) / 100
investment_amount_idr = st.sidebar.number_input("Investment Amount (IDR)", value=10000000, step=100000)
years_of_data = st.sidebar.number_input("Years of Data", min_value=1, max_value=20, value=5, step=1)
max_weight = st.sidebar.number_input("Maximum weight per asset (%)", min_value=1, max_value=100, value=100) / 100
min_weight = st.sidebar.number_input("Minimum weight per asset (%)", min_value=0, max_value=100, value=0) / 100
usd_price_idr = st.sidebar.number_input("Current USD Price (IDR)", value=16350, step=50)

# --- DATA FETCHING ---
end_date = datetime.today()
start_date = end_date - timedelta(days=years_of_data * 365)
adj_close_df = pd.DataFrame()
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]

if include_mutual_fund:
    tickers.append("Mutual Fund")

with st.spinner('Fetching historical data...'):
    for ticker in tickers:
        if ticker == "Mutual Fund":
            continue
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty and 'Adj Close' in data.columns:
                adj_close_df[ticker] = data['Adj Close']
            elif not data.empty and 'Close' in data.columns:
                 st.warning(f"No 'Adj Close' data for {ticker}. Using 'Close' price.")
                 adj_close_df[ticker] = data['Close']
            else:
                st.error(f"Could not fetch any price data for {ticker}.")
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

if include_mutual_fund:
    if not adj_close_df.empty:
        dates = adj_close_df.index
        start_value = 1000 
        daily_return_rate = (1 + mutual_fund_return)**(1/252) - 1
        daily_returns = np.full(shape=(len(dates),), fill_value=daily_return_rate)
        mutual_fund_prices = start_value * (1 + daily_returns).cumprod()
        adj_close_df["Mutual Fund"] = pd.Series(mutual_fund_prices, index=dates)
    elif "Mutual Fund" in tickers and len(tickers) == 1:
        st.error("Cannot run optimization with only a mutual fund. Please add other assets.")
        st.stop()
    else:
        st.error("No data available for other tickers. Cannot add mutual fund.")
        st.stop()

# --- MAIN LOGIC BLOCK ---
if adj_close_df.empty:
    st.error("No data available for the selected tickers. Please check your ticker symbols.")
else:
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

    if log_returns.empty:
        st.error("Not enough historical data for the selected timeframe. Please select a longer 'Years of Data' period.")
    else:
        cov_matrix = log_returns.cov() * 252

        if include_mutual_fund and "Mutual Fund" in cov_matrix.index:
            cov_matrix.loc["Mutual Fund", :] = 0
            cov_matrix.loc[:, "Mutual Fund"] = 0

        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        bounds = [(min_weight, max_weight)] * len(tickers)
        initial_weights = np.ones(len(tickers)) / len(tickers)

        if optimization_metric == "Sharpe Ratio":
            st.info("🚀 Optimizing portfolio for the best Sharpe Ratio...")
            objective_function = neg_sharpe_ratio
            args = (log_returns, cov_matrix, risk_free_rate_input)
        else: # Sortino Ratio
            st.info("🚀 Optimizing portfolio for the best Sortino Ratio...")
            objective_function = neg_sortino_ratio
            args = (log_returns, risk_free_rate_input)
        
        optimized_results = minimize(objective_function, initial_weights, args=args,
                                     method='SLSQP', constraints=constraints, bounds=bounds)

        optimal_weights = optimized_results.x
        capital_allocation_idr = optimal_weights * investment_amount_idr

        # --- CALCULATE METRICS ---
        portfolio_expected_return = expected_return(optimal_weights, log_returns) * 100
        portfolio_risk = standard_deviation(optimal_weights, cov_matrix) * 100
        portfolio_sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input)
        portfolio_sortino = sortino_ratio(optimal_weights, log_returns, risk_free_rate_input)
        portfolio_daily_returns = np.dot(log_returns, optimal_weights)
        cumulative_returns_series = pd.Series((1 + portfolio_daily_returns).cumprod(), index=log_returns.index)
        max_dd = max_drawdown(cumulative_returns_series)
        portfolio_var = value_at_risk(portfolio_daily_returns)
        portfolio_es = expected_shortfall(portfolio_daily_returns, portfolio_var)

        # --- DISPLAY RESULTS ---
        st.subheader('📝 Asset Details & Allocation')
        assets_data = []
        with st.spinner('Fetching current prices for table...'):
            for i, ticker in enumerate(tickers):
                weight = optimal_weights[i]
                capital = capital_allocation_idr[i]
                current_price = get_current_price(ticker)

                if current_price is not None:
                    currency = "USD" if "-USD" in ticker else "IDR"
                    price_str = f"{current_price:,.4f} {currency}" if currency == "USD" else f"Rp {current_price:,.0f}"

                    if ticker == "Mutual Fund":
                        shares_str = "N/A"
                        price_str = "N/A"
                    elif currency == "USD":
                        shares = capital / (current_price * usd_price_idr)
                        shares_str = f"{shares:.8f}"
                    else: # IDR Stocks
                        shares = np.floor(capital / current_price / 100) * 100 if current_price > 0 else 0
                        shares_str = f"{int(shares)}"
                    
                    assets_data.append([ticker, f"{weight*100:.2f}%", f"Rp {capital:,.0f}", price_str, shares_str])
                else:
                    assets_data.append([ticker, f"{weight*100:.2f}%", f"Rp {capital:,.0f}", "Price not found", "N/A"])
            
        assets_df = pd.DataFrame(assets_data, columns=["Asset", "Weighting", "Allocated Capital", "Current Price", "Shares"])
        st.dataframe(assets_df)

        st.subheader('📊 Portfolio Metrics')
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""<div class="metric-card metric-return"><div class="icon">📈</div><h3>Expected Return</h3><p class="value">{portfolio_expected_return:.2f}%</p></div>""", unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""<div class="metric-card metric-risk"><div class="icon">⚖️</div><h3>Risk (Std Dev)</h3><p class="value">{portfolio_risk:.2f}%</p></div>""", unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""<div class="metric-card metric-sharpe"><div class="icon">📊</div><h3>Sharpe Ratio</h3><p class="value">{portfolio_sharpe:.2f}</p></div>""", unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""<div class="metric-card metric-other"><div class="icon">⚡</div><h3>Sortino Ratio</h3><p class="value">{portfolio_sortino:.2f}</p></div>""", unsafe_allow_html=True)

        m_col5, m_col6, m_col7 = st.columns(3)
        with m_col5:
            st.markdown(f"""<div class="metric-card metric-drawdown"><div class="icon">⛔</div><h3>Max Drawdown</h3><p class="value">{max_dd * 100:.2f}%</p></div>""", unsafe_allow_html=True)
        with m_col6:
            st.markdown(f"""<div class="metric-card metric-other"><div class="icon">💥</div><h3>Value at Risk (VaR)</h3><p class="value">{portfolio_var * 100:.2f}%</p></div>""", unsafe_allow_html=True)
        with m_col7:
            st.markdown(f"""<div class="metric-card metric-other"><div class="icon">💸</div><h3>Expected Shortfall (ES)</h3><p class="value">{portfolio_es * 100:.2f}%</p></div>""", unsafe_allow_html=True)

        st.subheader('📈 Performance and Allocation Visuals')
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.write('**Portfolio Weights Distribution**')
            # Doughnut chart for better aesthetics
            fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
            ax_pie.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4), pctdistance=0.8)
            ax_pie.set_title('Optimal Portfolio Weights', fontsize=14, fontweight='bold')
            st.pyplot(fig_pie)
            plt.close(fig_pie) # --- FIX: Close the figure ---

            st.write('**Capital Allocation (IDR)**')
            fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
            ax_bar.bar(tickers, capital_allocation_idr)
            ax_bar.set_ylabel('Allocated Capital (IDR)')
            ax_bar.set_title('Capital Allocation', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig_bar)
            plt.close(fig_bar) # --- FIX: Close the figure ---

        with g_col2:
            st.write('**Portfolio Cumulative Returns**')
            fig_line, ax_line = plt.subplots(figsize=(10, 8))
            ax_line.plot(cumulative_returns_series.index, cumulative_returns_series.values, label='Optimized Portfolio', color="#4CAF50", linewidth=2)
            ax_line.set_title('Portfolio Performance Over Time', fontsize=14, fontweight='bold')
            ax_line.set_xlabel('Date')
            ax_line.set_ylabel('Cumulative Returns')
            plt.xticks(rotation=45, ha="right")
            ax_line.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_line)
            plt.close(fig_line) # --- FIX: Close the figure ---
