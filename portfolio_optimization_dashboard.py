import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Portfolio Optimization Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #2F4F4F;
        }
        .subheader {
            font-size: 24px;
            color: #4CAF50;
            font-weight: 600;
        }
        .metrics {
            font-size: 18px;
            margin-bottom: 10px;
            font-weight: 500;
        }
        .metric-value {
            color: #1E90FF;
            font-weight: 600;
        }
        .streamlit-expanderHeader {
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Title with styled text
st.markdown('<p class="title">📈 Portfolio Optimization Dashboard</p>', unsafe_allow_html=True)

# Description
st.markdown("""
    Optimize your investment portfolio by allocating capital across various assets.
    You can view portfolio statistics, including expected return, risk, Sharpe ratio, and more. 
    Adjust settings and watch the magic happen!  
    **Optimizing for the best risk-adjusted returns.**
    """, unsafe_allow_html=True)

# Sidebar with custom logo and input widgets
st.sidebar.image("Designer.png", use_container_width=True)  
st.sidebar.header("Portfolio Inputs")

# Sidebar elements
tickers_input = st.sidebar.text_input("Enter Asset Tickers (e.g., BBCA.JK, BTC-USD, TSLA)", "BBCA.JK, BTC-USD")
risk_free_rate_input = st.sidebar.number_input("Risk-Free Rate (%)", value=6.0, step=0.1) / 100  # Decimal format
investment_amount_idr = st.sidebar.number_input("Investment Amount (IDR)", value=10000000, step=100000)
years_of_data = st.sidebar.number_input("Years of Data", min_value=1, max_value=20, value=5, step=1)
max_weight = st.sidebar.number_input("Maximum Weight Per Asset (%)", min_value=1, max_value=100, value=50) / 100
min_weight = st.sidebar.number_input("Minimum Weight Per Asset (%)", min_value=0, max_value=100, value=0) / 100
usd_price_idr = st.sidebar.number_input("Current USD Price (IDR)", value=15000, step=100)

# Load data and perform calculations
# Fetch adjusted close price data for each ticker
# ... (same as your original code)

# Display optimal portfolio results
if adj_close_df.empty:
    st.error("No data available for the selected tickers.")
else:
    # Display Portfolio Optimization Result in an Interactive Way
    st.subheader("Optimized Portfolio Allocation")
    portfolio_df = pd.DataFrame({
        'Asset': tickers,
        'Weight': optimal_weights,
        'Allocated Capital (IDR)': capital_allocation_idr,
        'Amount of Shares': shares
    })

    # Use streamlit's interactive table with more refined display options
    st.dataframe(portfolio_df.style.format({
        'Allocated Capital (IDR)': "Rp {:,.2f}",
        'Weight': "{:.4f}",
        'Amount of Shares': "{:.8f}"
    }), use_container_width=True)

    # Portfolio Metrics Section
    st.subheader("Portfolio Performance Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"📊 **Expected Return (Annualized)**: <span class='metric-value'>{portfolio_expected_return:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"📉 **Risk (Standard Deviation)**: <span class='metric-value'>{portfolio_risk:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"📈 **Sharpe Ratio**: <span class='metric-value'>{sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate_input):.2f}</span>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"📊 **Sortino Ratio**: <span class='metric-value'>{portfolio_sortino:.2f}</span>", unsafe_allow_html=True)
        st.markdown(f"⚠️ **Maximum Drawdown**: <span class='metric-value'>{max_dd * 100:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"📉 **Value-at-Risk (VaR) at 95% Confidence**: <span class='metric-value'>{portfolio_var * 100:.2f}%</span>", unsafe_allow_html=True)
        st.markdown(f"📉 **Expected Shortfall (ES) at 95% Confidence**: <span class='metric-value'>{portfolio_es * 100:.2f}%</span>", unsafe_allow_html=True)

    # Graphs in separate sections
    st.subheader("Portfolio Visualizations")

    # Cumulative Returns Graph
    st.markdown("### Cumulative Returns of the Portfolio")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cumulative_returns, label='Optimized Portfolio', color="#4CAF50", linewidth=2)
    ax.set_title("Portfolio Cumulative Returns", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (Days)", fontsize=12)
    ax.set_ylabel("Cumulative Returns", fontsize=12)
    ax.legend()
    st.pyplot(fig)

    # Portfolio Weights Distribution (Pie chart)
    st.markdown("### Portfolio Weights Distribution")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title("Optimal Portfolio Weights", fontsize=14, fontweight='bold')
    st.pyplot(fig)

    # Capital Allocation in IDR (Bar chart)
    st.markdown("### Capital Allocation in IDR")
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(tickers, capital_allocation_idr, color=plt.cm.Paired.colors)
    ax.set_xlabel('Assets', fontsize=12)
    ax.set_ylabel('Allocated Capital (Rp)', fontsize=12)
    ax.set_title("Capital Allocation for Investment", fontsize=14, fontweight='bold')

    # Add annotations for capital amounts
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 50000, f'Rp {height:,.2f}', 
                 ha='center', va='bottom', fontsize=10, color='black')

    st.pyplot(fig)

    # Export data as CSV/Excel (button feature)
    st.download_button(
        label="Download Portfolio Data (CSV)",
        data=portfolio_df.to_csv(index=False),
        file_name="optimized_portfolio.csv",
        mime="text/csv"
    )
