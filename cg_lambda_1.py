import numpy as np
import matplotlib.pyplot as plt
from Utilities_NEW import dendrogram_plot, dendrogram_plot_test
import pandas as pd
import glob

market_list = ["all"]

for i in range(len(market_list)):
    market_period_i = market_list[i]
    market_period = market_period_i # all, gfc, gfc_crash, interim, covid, covid_crash

    # plot parameters
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    labels = ["10,1","10,2","10,3","10,4",
              "1,1","1,2","1,3","1,4",
            "2,1", "2,2","2,3","2,4",
            "3,1","3,2","3,3", "3,4",
             "4,1", "4,2","4,3","4,4",
              "5,1","5,2","5,3","5,4",
              "6,1","6,2","6,3","6,4",
              "7,1","7,2","7,3","7,4",
              "8,1","8,2","8,3","8,4",
              "9,1","9,2","9,3","9,4"] # Try 10 at the top

    # Read in data
    path = '/Users/tassjames/Desktop/crypto_mdpi/lambda_paths' # use your path
    all_files = glob.glob(path + "/*.csv")
    all_files.sort()

    lamda_1_list = []
    lamda_means_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_slice = np.array(df.iloc[1,1:])
        lamda_1_list.append(df_slice)

    for j in range(len(lamda_1_list)):
        lamda_means_list.append(np.mean(df_slice))

    # Convert to an array
    distance_matrix = np.zeros((len(lamda_1_list),len(lamda_1_list)))
    for i in range(len(lamda_1_list)):
        for j in range(len(lamda_1_list)):
            if market_period == "all":
                lamda_1_i = lamda_1_list[i]
                lamda_1_j = lamda_1_list[j]
                # Compute L1 distance between vectors
                # dist = np.sum(np.abs(lamda_1_i - lamda_1_j))/len(lamda_1_i)
                dist = np.sum(np.abs(lamda_1_i - lamda_1_j))
                distance_matrix[i,j] = dist

        print("Iteration", i)

    # Dendrogram plot labels
    #dendrogram_plot_test(distance_matrix, "_L1_", "Z_lambda_1_L1_crypto_"+market_period_i, labels)
    #print(np.linalg.norm(distance_matrix))