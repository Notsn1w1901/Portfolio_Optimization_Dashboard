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
st.set_page_config(page_title="Portfolio Optimization", layout="wide", initial_sidebar_state="expanded")

# Apply custom CSS
st.markdown("""
    <style>
    /* Custom CSS for Streamlit Dashboard */
    
    /* General Styling */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f7fc;
    }
    
    /* Title Styling */
    h1 {
        font-size: 2.5rem;
        color: #2C3E50;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #34495E;
        color: #ECF0F1;
    }

    .sidebar .sidebar-header {
        background-color: #2C3E50;
        color: #ECF0F1;
    }

    .sidebar .sidebar-header h2 {
        font-size: 1.5rem;
        font-weight: 600;
    }

    .stButton>button {
        background-color: #16A085;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
    }

    .stButton>button:hover {
        background-color: #1ABC9C;
    }

    /* Graph Styling */
    .stImage, .stPlot {
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }

    /* DataFrame Table Styling */
    .dataframe {
        border-collapse: collapse;
    }

    .dataframe th, .dataframe td {
        padding: 12px 20px;
        border: 1px solid #ddd;
    }

    .dataframe th {
        background-color: #2C3E50;
        color: #ECF0F1;
    }

    .dataframe td {
        text-align: center;
        color: #2C3E50;
    }

    /* Subheader Styling */
    h3, .stSubheader {
        font-size: 1.5rem;
        color: #2C3E50;
        margin-bottom: 10px;
        font-weight: 600;
    }

    /* Buttons & Inputs */
    .stButton>button, .stTextInput>input, .stNumberInput>input {
        font-size: 1rem;
        color: #2C3E50;
        border-radius: 5px;
        padding: 8px;
    }

    /* Bar Chart Styling */
    .bar {
        border-radius: 8px;
        padding: 10px;
        background-color: #ecf0f1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stSelectbox, .stTextInput, .stNumberInput {
        font-size: 1.1rem;
        padding: 8px;
        border-radius: 5px;
        background-color: #fff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stSelectbox:hover, .stTextInput:hover, .stNumberInput:hover {
        background-color: #f1f1f1;
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
    and display the expected return, risk (standard deviation), and capital allocation for each asset in both IDR and USD.
""", unsafe_allow_html=True)

# Your code for the rest of the app continues here...
