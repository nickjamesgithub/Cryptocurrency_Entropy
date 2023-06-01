import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

# Import data
prices = pd.read_csv("/Users/tassjames/Desktop/crypto_mdpi/final_data_cleaned_sizes.csv", index_col='Date')
market = (prices.iloc[2:, 1:]).astype("float") # Get market prices
market_returns_ = np.log(market).diff() # Compute log returns of market
market_returns_.columns = market_returns_.columns.str.replace('(\.\d+)$', '')

# Model parameters
smoothing_rate = 120
corr_1 = [] # Explanatory variance of first eigenvalue list
inner_product_ones = [] # inner Product ones

# Store sector pathways
sector_pathways_ip_list = []
sector_pathways_corr_list = []

for i in range(smoothing_rate, len(market_returns_)-1): # len(market_returns)
    print(i)
    # Take market correlation matrix
    market_returns_slice = market_returns_.iloc[(i - smoothing_rate):i, :]
    market_correlation = np.nan_to_num(market_returns_slice.corr())  # Compute correlation matrix
    m_vals, m_vecs = eigsh(market_correlation, k=len(market_correlation), which='LM')
    list = []
    m_vals_e = np.reshape((m_vals)[-1], (1,1))
    m_vecs_e = np.reshape((m_vecs)[:,-1], (len(market_correlation),1))
    Av = np.dot(market_correlation, m_vecs_e)
    lamdaV = np.dot(m_vecs_e, m_vals_e)
    error = max(max(abs(Av - lamdaV)))
    m_vecs = m_vecs[:, -1]  # Get 1st eigenvector
    m_vals_1 = m_vals[-1]/len(market_correlation)
    corr_1.append(m_vals_1)

    # Inner product with ones
    one_vector = np.ones(len(market_correlation))/len(market_correlation)
    norm_inner_product_ones = np.dot(m_vecs, one_vector)/(np.linalg.norm(m_vecs) * np.linalg.norm(one_vector))
    norm_1 = np.linalg.norm(m_vecs)
    norm_2 = np.linalg.norm(one_vector)
    # norm_inner_product_ones = np.abs(np.dot(s_vecs, one_vector)/(np.linalg.norm(s_vecs) * np.linalg.norm(one_vector))) # Inner product
    inner_product_ones.append(norm_inner_product_ones) # Norm Inner product between (111...1) and first eigenvector of sector

    # Append m_vals_1 and inner product
    sector_pathways_corr_list.append(m_vals_1)
    sector_pathways_ip_list.append(norm_inner_product_ones)

    print("Simulation " + str(i))

# Split up into partitions
ip_split_array = np.array(sector_pathways_ip_list)
corr_split_array = np.array(sector_pathways_corr_list)

# Date index
date_index_plot = pd.date_range('10-01-2019','14-02-2023',1202).strftime('%Y-%m-%d')

# Normalized eigenvalue 1
fig, ax = plt.subplots()
plt.plot(date_index_plot, np.abs(ip_split_array))
plt.tick_params(axis='x', which='major', labelsize=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
plt.ylim(0.25,1)
plt.savefig("Crypto_G_Market_IP_1_all")
plt.show()

# Inner product ones #todo 1
fig, ax = plt.subplots()
plt.plot(date_index_plot, corr_split_array)
plt.tick_params(axis='x', which='major', labelsize=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
plt.ylim(0.25,1)
plt.savefig("Crypto_G_Explanatory_variance_eigenvalue_1_all")
plt.show()






