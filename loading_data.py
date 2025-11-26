import os
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred
from pytrends.request import TrendReq

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")  # Your FRED API key


START_DATE = "2018-01-02"  # S&P 500 full data starts here
END_DATE = "2025-10-01"


print("Downloading S&P 500 data...")
ticker = "^GSPC"
sp500 = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1d")
sp500.index = pd.to_datetime(sp500.index)
print("S&P 500 data loaded:", sp500.shape)


fred = Fred(api_key=FRED_API_KEY)
indicators = {
    "CPI": "CPIAUCSL",
    "Unemployment": "UNRATE",
    "FedFunds": "FEDFUNDS"
}

print("Downloading FRED macroeconomic data...")
macro_data = pd.DataFrame()
for name, code in indicators.items():
    series = fred.get_series(code, observation_start=START_DATE)
    series = series.rename(name)
    macro_data = pd.concat([macro_data, series], axis=1)

macro_data.index = pd.to_datetime(macro_data.index)
macro_data = macro_data.resample('D').ffill()  # Forward-fill missing days
print("Macro data loaded:", macro_data.shape)


print("Downloading Google Trends data...")
pytrends = TrendReq(hl='en-US', tz=360)
search_terms = ["recession", "stock market crash", "inflation"]
pytrends.build_payload(search_terms, timeframe=f'{START_DATE} {END_DATE}', geo='US')
trend_data = pytrends.interest_over_time()

if 'isPartial' in trend_data.columns:
    trend_data = trend_data.drop(columns='isPartial')

trend_data.index = pd.to_datetime(trend_data.index)
trend_data = trend_data.resample('D').ffill()  # Forward-fill missing days
print("Google Trends data loaded:", trend_data.shape)


print("Merging datasets...")
full_data = sp500.join([macro_data, trend_data], how="outer")
full_data = full_data.loc[START_DATE:]  # Keep only data from 2018-01-02 onward
full_data = full_data.ffill()  # Fill any remaining NaNs
print("Merged dataset shape:", full_data.shape)


print("Computing engineered features...")

# Flatten multi-index columns from yfinance
if isinstance(full_data.columns, pd.MultiIndex):
    full_data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in full_data.columns]

# Identify the S&P 500 Close column dynamically
close_col_candidates = [c for c in full_data.columns if "Close" in c]
if not close_col_candidates:
    raise KeyError("Cannot find the S&P 500 Close column in your dataset.")
close_col = close_col_candidates[0]
print("Using Close column:", close_col)

# Compute features
full_data['LogReturn'] = np.log(full_data[close_col] / full_data[close_col].shift(1))
full_data['Volatility_10d'] = full_data['LogReturn'].rolling(window=10).std()
full_data['Momentum_10d'] = full_data[close_col] - full_data[close_col].shift(10)
full_data['SMA_10'] = full_data[close_col].rolling(window=10).mean()
full_data['SMA_50'] = full_data[close_col].rolling(window=50).mean()
full_data['EMA_10'] = full_data[close_col].ewm(span=10, adjust=False).mean()
full_data['EMA_50'] = full_data[close_col].ewm(span=50, adjust=False).mean()
rolling_max = full_data[close_col].cummax()
full_data['Drawdown'] = (full_data[close_col] - rolling_max) / rolling_max

# Drop initial NaNs from rolling calculations
full_data = full_data.dropna()


full_data.to_csv("sp500_features.csv")
print("✅ Feature dataset saved as sp500_features.csv")
print(full_data.head())


