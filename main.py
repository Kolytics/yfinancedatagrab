#!/usr/bin/env python
# coding: utf-8

# In[735]:


import yfinance as yf
import pandas as pd
import numpy as np
import math
import time


# In[879]:


def fetch_max_history(ticker: str) -> pd.DataFrame:
    """Using yahoo finance, fetch max pricing history.
    Export to file and also return the data."""
    # Historical and adjusted
    yf_stock = yf.Ticker(ticker)
    time.sleep(1)
    yf_history = yf_stock.history(period="max")
    time.sleep(1)
    yf_history.to_csv(f"data/adjusted/{ticker}.csv")
    # unadjusted data:
    unadjusted_history = yf.download([ticker], period="max")
    unadjusted_history.to_csv(f"data/unadjusted/{ticker}.csv")
    return yf_history


# In[880]:


def fetch_all_stocks(stocks: pd.DataFrame) -> None:
    """Iterates through list of stocks dataframe
    and fetches max history"""
    for i, v in stocks.iterrows():
        print("Processing: ", i, v['Ticker'])
        fetch_max_history(v['Ticker'])
        time.sleep(3)


# In[881]:


def cumulative_returns(returns):
    res = (returns + 1.0).cumprod()
    return res


# In[878]:


# Load the list of stock data from file.
stocks = pd.read_csv("stocks.csv")


# In[882]:


# Download each stock into an individual CSV
fetch_all_stocks(stocks)


# In[883]:


# Load the stocks into a list and add the index=FTSE100
list_stocks = list(stocks['Ticker']) # Convert the pandas to a list of tickers.


# In[884]:


# Download and export all of stock data
all_stocks = yf.download(list_stocks , start="1980-01-01") # Download the data
all_stocks.to_csv('data/all_stocks.csv')


# In[885]:


def get_prices_between_dates(frame: pd.DataFrame,
                             start_date: str,
                             end_date: str) -> pd.DataFrame:
    """Get the prices between two dates. The dates are inclusive."""
    mask = (all_stocks.index >= start_date) & (all_stocks.index <= end_date)
    return frame.loc[mask]


# In[886]:


def clean_stock_matrix(stocks: pd.DataFrame,
                       asset: str = None,
                       freq: str = 'daily') -> pd.DataFrame:
    µ = stocks['Adj Close']

    # If we have NaNs, find the first valid index
    if asset != None:
        if math.isnan(µ[asset][0]):
            µ = µ.loc[µ.index >= µ[asset].first_valid_index()]

    if freq=='monthly':
        µ = µ.resample('BM').apply(lambda x: x[-1]).pct_change()
    elif freq=='annual':
        µ = µ.resample('BA').apply(lambda x: x[-1]).pct_change()
    else:
        µ = µ.pct_change()
    
    return µ


# In[887]:


def slope(stocks: pd.DataFrame,
          asset1: str, # should be e.g. ^FTSE
          asset2: str,
          freq: str = 'daily') -> np.ndarray:
    """Get the Alpha and Beta for two assets by applying linear regression."""
    µ = clean_stock_matrix(stocks, freq=freq, asset=asset2)

    X = np.array(µ[asset1][1:])
    y = np.array(µ[asset2][1:])
    X_b = np.c_[np.ones(X.shape), X] # set bias term to 1 for each sample
    # normal equationn(X.T * X)^(-1) * X.T * y 
    return np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


# In[888]:


def covariance(stocks, asset1: str = None, asset2: str = None, freq: str ='daily'):
    """Calculate the covariance for two assets."""
    µ = clean_stock_matrix(stocks, freq=freq, asset=asset1)
    cov = µ[1:].cov()
    if asset1 != None and asset2 != None:
        return cov[asset1][asset2]
    else:
        return cov
    
def calculate_beta_using_cov(cov, cov_matrix: pd.DataFrame, stock1, stock2):
    return cov[stock1][stock2]/np.var(cov_matrix['Adj Close'][stock2].pct_change()[1:], ddof=1)


# # Load all stock data from CSV
# * **input**: start_date, end_date, freq = monthly or daily
# * **output**: DataFrame with β, α, and cov for each of the stocks

# In[889]:


# Load the data
# Data should be c.14mb
loaded_stocks = pd.read_csv('data/all_stocks.csv', low_memory=False)


# In[890]:


# Get a specific date range
# The index of the DataFrame is the Date
stock_range1 = get_prices_between_dates(all_stocks, '2016-06-01', '2021-06-14')


# In[891]:


# Apply the regression and get the coeff for a specific stock
params = slope(stock_range1, '^FTSE', 'SGRO.L', freq='annual')
print('Alpha: ', params[0])
print('Beta: ', params[1])


# In[ ]:





# ## Get β, α, and covariance for each of the stocks

# In[842]:


def df_beta(row):
    params = slope(stock_range1, '^FTSE', row.Ticker, freq='daily')
    return params[1]

def df_alpha(row):
    params = slope(stock_range1, '^FTSE', row.Ticker, freq='daily')
    return params[0]


# In[892]:


indices = ['^FTSE', '^FTLC', 'IUKP.L']
rangex = ['daily', 'monthly', 'annual']


# In[893]:


new_stocks = stocks
for r in rangex:
    cov = covariance(stock_range1, freq=r)
    cov.to_csv(f'data/cov_{r}_range.csv')
    for index in indices:
        new_stocks[f'Beta ({index})'] = stocks.apply(lambda row: slope(stock_range1, index, row.Ticker, freq=r)[1], axis = 1)
        new_stocks[f'Alpha ({index})'] = stocks.apply(lambda row: slope(stock_range1, index, row.Ticker, freq=r)[0], axis = 1)
        new_stocks[f'Covariance ({index})'] = stocks.apply(lambda row: cov[row.Ticker][index], axis = 1)
    new_stocks.to_csv(f'data/{r}_range_stocks.csv')


# 
# # Testing with GOOG and SPY

# In[484]:


# Testing with GOOG and SPY
# Download the data
g = yf.download(['GOOG', 'BLND.L', 'SPY'] , start="2016-06-30") # Download the data


# In[631]:


# Apply the regression and get the coeffs
params = slope(g, 'BLND.L', 'SPY', freq='daily')
print('Alpha: ', params[0])
print('Beta: ', params[1])


# # Covariance

# In[796]:


def apply_cov_1x1(row):
    return covariance(stock_range1, row.Ticker, '^FTSE', freq='daily')
stocks['Covariance'] = stocks.apply(apply_cov_1x1, axis = 1)

