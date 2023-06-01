import numpy as np
import matplotlib.pyplot as plt
from Utilities_NEW import dendrogram_plot, dendrogram_plot_test
import pandas as pd
import glob

market_list = ["all"]

for i in range(len(market_list)):
    market_period_i = market_list[i]
    market_period = market_period_i # all, gfc, gfc_crash, interim, covid, covid_crash

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
    lamda_1_mean_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_slice = np.array(df.iloc[1,1:])
        lamda_1_list.append(df_slice)
        lamda_1_mean_list.append([np.mean(df_slice)])

    # Lambda 1 mean + labels
    combined = lamda_1_mean_list + labels