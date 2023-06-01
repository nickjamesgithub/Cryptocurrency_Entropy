import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import wasserstein_distance
from scipy.sparse import linalg

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/Diffusion_maps_financial/sp500_clean_labels_sectors.csv", index_col='Date')
prices.dropna(axis='columns', inplace=True)

# Automatically  rename sector labels
sectors_labels = ["#N/A", "Health Care", "Industrials", "Communication Services", "Information Technology", "Utilities", "Financials",
           "Materials", "Real Estate", "Consumer Staples", "Consumer Discretionary", "Energy"]
sectors_labels.sort()

# Replace column names for prices and returns
prices = prices.reindex(sorted(prices.columns), axis=1)
prices.columns = prices.columns.str.replace('(\.\d+)$','')
prices_df = pd.DataFrame(prices)
prices_df.to_csv("/Users/tassjames/Desktop/prices_sector_labels.csv")
