import numpy as np
import pandas as pd
from scipy.stats import multinomial
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.linalg import eigsh

make_plots = False

# Choose number of sectors and n for simulation
sectors_list = [1,2,3,4,5,6,7,8,9,10] # 2,3,4,5,6,7,8,9,10
samples_list = [1,2,3,4] # 2,3,4,5,6,7,8,9

for k in range(len(sectors_list)):
    for s in range(len(samples_list)):

        # Import data
        prices = pd.read_csv("/Users/tassjames/Desktop/crypto_mdpi/final_data_cleaned_sizes.csv", index_col='Date')

        num_sectors = sectors_list[k]
        sample_per_sector = samples_list[s]

        # n is samples per sector * number of sectors
        n = sample_per_sector * num_sectors

        num_simulations = 500 # 500

        # Replace column names for prices
        prices = prices.reindex(sorted(prices.columns), axis=1)
        prices.columns = prices.columns.str.replace('(\.\d+)$','')

        sectors_labels = ["1","2", "3", "4", "5", "6", "7", "8", "9", "10"]

        sectors_labels.sort()

        first_eigenvalue_samples = []
        portfolio_returns_sample = []
        while len(first_eigenvalue_samples) < num_simulations:
            # First pick n sectors at random
            sector_sequence = list(np.linspace(0,len(sectors_labels)-1,len(sectors_labels))) # Randomly draw sector numbers
            random_list_sector = random.sample(sector_sequence, num_sectors)
            ints = [int(item) for item in random_list_sector]

            # Get corresponding sector names
            random_sector_list = []
            for i in range(len(ints)):
                random_sector_drawn = sectors_labels[ints[i]]
                random_sector_list.append(random_sector_drawn)

            # Print random sector list
            print(random_sector_list)

            # Get the random samples for current iteration
            stock_samples = []
            names = []
            for i in range(len(random_sector_list)):
                sector_slice = prices[random_sector_list[i]]
                length = len(sector_slice.columns)
                random_sequence = list(np.linspace(0, length - 1, length))
                random_list_stocks = random.sample(random_sequence, sample_per_sector)
                ints = [int(item) for item in random_list_stocks]
                random_sector_stocks = sector_slice.iloc[:, ints]
                for j in range(len(random_sector_stocks.iloc[0])):
                    stock_slice = random_sector_stocks.iloc[:, j]
                    stock_slice_list = list((stock_slice[1:]).astype("float"))
                    names.append(stock_slice[0])
                    stock_samples.append(stock_slice_list)

            # Convert back into a dataframe
            stock_samples_df = pd.DataFrame(np.transpose(stock_samples))
            log_returns = np.log(stock_samples_df).diff()[1:]
            smoothing_rate = 90

            returns_list = []
            corr_1 = []
            for i in range(smoothing_rate, len(log_returns)):
                # Returns
                returns = log_returns.iloc[i - smoothing_rate:i, :]
                # Compute with pandas
                correlation = np.nan_to_num(returns.corr())

                # Perform eigendecomposition and get explanatory variance
                m_vals, m_vecs = eigsh(correlation, k=6, which='LM')
                m_vecs = m_vecs[:, -1]  # Get 1st eigenvector
                m_vals_1 = m_vals[-1] / len(correlation)
                corr_1.append(m_vals_1)
                print("Sectors ", sectors_list[k], " Samples", samples_list[s])
                print("Iteration "+str(i)+" / "+str(len(log_returns)))

                # Compute total returns
                returns_1 = np.array(log_returns.iloc[i, :])
                weights = np.repeat(1/len(returns_1), n)
                total_return_iteration = np.sum(returns_1 * weights)
                returns_list.append(total_return_iteration)

            # Append draws of first eigenvalue samples to main list
            first_eigenvalue_samples.append(corr_1)
            portfolio_returns_sample.append(returns_list)
            print("Simulation " + str(len(first_eigenvalue_samples)) + " / " + str(num_simulations))

        # Generate average \lambda_1(t) sample path and confidence intervals
        first_eigenvalue_array = np.array(first_eigenvalue_samples)
        lambda_sample_90 = np.percentile(first_eigenvalue_array, 95, axis=0)
        lambda_sample_50 = np.percentile(first_eigenvalue_array, 50, axis=0)
        lambda_sample_5 = np.percentile(first_eigenvalue_array, 5, axis=0)
        lambda_paths = [lambda_sample_5, lambda_sample_50, lambda_sample_90]

        # Plot eigenvalue samples at 5th, 50th and 95th percentile
        if make_plots:
            plt.plot(lambda_sample_5, label="5th percentile")
            plt.plot(lambda_sample_50, label="50th percentile")
            plt.plot(lambda_sample_90, label="95th percentile")
            plt.show()

        # Print lambda pathways
        print(lambda_paths)

        # # Portfolio return samples
        # portfolio_volatilities = []
        # for i in range(len(portfolio_returns_sample)):
        #     volatility = np.std(portfolio_returns_sample[i]) * np.sqrt(250)
        #     portfolio_volatilities.append(np.nan_to_num(volatility))
        # print(portfolio_volatilities)

        # Convert to dataframes and write to csv
        # portfolio_volatilities_df = pd.DataFrame(portfolio_volatilities)
        # portfolio_volatilities_df.to_csv("/Users/tassjames/Desktop/jacob_financial_crises/volatilities"+"_"+crisis[c]+"_"+str(num_sectors)+'_'+str(sample_per_sector)+".csv")
        lambda_paths_df = pd.DataFrame(lambda_paths)
        lambda_paths_df.to_csv("/Users/tassjames/Desktop/crypto_mdpi/lambda_paths/"+str(num_sectors)+'_'+str(sample_per_sector)+".csv")
