from scipy.stats.kde import gaussian_kde
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from sklearn.metrics import mutual_info_score
import scipy.cluster.hierarchy as sch
import pylab
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import squareform


def coefficient_distribution_plot(values_array, period):
    # Plot standard histogram
    plt.hist(values_array, bins=100, alpha=0.25)
    # Plot Kernel Density Estimate
    kde = gaussian_kde(values_array)
    dist_space = np.linspace(np.min(values_array), np.max(values_array), len(values_array))
    plt.plot(dist_space, kde(dist_space))
    plt.xlabel("Correlation coefficients")
    plt.ylabel("Frequency")
    plt.title(period+" Correlation Coefficient")
    # plt.savefig("GFC_correlation_coefficients")
    plt.show()

def flatten_dataframe(dataframe):
    df_out = dataframe.corr().stack()
    df_out = df_out[df_out.index.get_level_values(0) != df_out.index.get_level_values(1)]
    df_out.index = df_out.index.map('_'.join)
    df_out = df_out.to_frame().T
    return df_out

def entropy(p):
    return -(p * np.log2(p) + (1-p) * np.log2((1-p)))

def dendrogram_plot(matrix, distance_measure, data_generation, labels):

    # Compute and plot dendrogram.
    plt.rcParams.update({'font.size': 20})
    fig = pylab.figure(figsize=(15,10))
    axdendro = fig.add_axes([0.09,0.1,0.2,0.8]) # 0.1, 0.1, 0.2, 0.8
    Y = sch.linkage(matrix, method='centroid')
    # Z = sch.dendrogram(Y, orientation='right', labels=labels, leaf_rotation=360, leaf_font_size=7)
    Z = sch.dendrogram(Y, orientation='right', leaf_rotation=360, leaf_font_size=9) # no_labels=True
    axdendro.set_xticks([])
    # axdendro.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = matrix[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)
    plt.savefig(data_generation+distance_measure+"Dendrogram",bbox_inches="tight")
    plt.show()

    # Display and save figure.
    fig.show()

def rebase(prices):
    return prices/prices[0]*100

def rank_data(sorted_idx):
    # Ranks
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(len(sorted_idx))
    return ranks[sorted_idx]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def mj1_distance(x,y):
    set_x = []
    set_y = []
    for i in range(len(x)):
        set_x.append(np.abs(x[i] - find_nearest(y, x[i])))
    for j in range(len(y)):
        set_y.append(np.abs(y[j] - find_nearest(x, y[j])))
    set_x = np.array(set_x)
    set_y = np.array(set_y)
    dist_vector = np.concatenate((set_x, set_y))
    mj1_distance = ((np.mean(set_x) + np.mean(set_y)) / 2)  # Reference 17
    return mj1_distance

# MJ-Wasserstein Code
def mj_wasserstein(ts1_cutpoints, ts1_probabilities, ts1_max_probability, ts2_cutpoints, ts2_probabilities, ts2_max_probability):

    # industrials is TS1
    # Communications is TS2
    values_x = []
    indices_x = []
    probabilities_distributions_x = []
    wasserstein_score_x = []

    values_y = []
    indices_y = []
    probabilities_distributions_y = []
    wasserstein_score_y = []

    for i in range(len(ts1_max_probability)):
        # Determine Nearest point and surrounding uncertainties in X
        nearest_values_x = find_nearest(ts2_max_probability, ts1_max_probability[i])[0]
        index_x = ts2_max_probability.index(nearest_values_x)
        prob_dist_x = ts2_probabilities[index_x]
        prob_dist_y = ts1_probabilities[i]
        cutpoints_x = ts2_cutpoints[index_x]
        cutpoints_y = ts1_cutpoints[i]

        # Compute MJ Wasserstein
        wasserstein_score = wasserstein_distance(cutpoints_x, cutpoints_y, prob_dist_x, prob_dist_y)
        wasserstein_score_x.append(wasserstein_score)

        # Append values to X
        values_x.append(nearest_values_x)
        indices_x.append(index_x)
        probabilities_distributions_x.append(prob_dist_x)

    for i in range(len(ts2_max_probability)):
        # Determine Nearest point and surrounding uncertainties in X
        nearest_values_y = find_nearest(ts1_max_probability, ts2_max_probability[i])[0]
        index_y = ts1_max_probability.index(nearest_values_y)
        prob_dist_x = ts1_probabilities[index_y]
        prob_dist_y = ts2_probabilities[i]
        cutpoints_x = ts1_cutpoints[index_y]
        cutpoints_y = ts2_cutpoints[i]

        # Compute MJ Wasserstein
        wasserstein_score = wasserstein_distance(cutpoints_x, cutpoints_y, prob_dist_x, prob_dist_y)
        wasserstein_score_y.append(wasserstein_score)

        # Append values to Y
        values_y.append(nearest_values_y)
        indices_y.append(index_y)
        probabilities_distributions_y.append(prob_dist_y)

    # MJ-Wasserstein Computation
    set_x = np.array(wasserstein_score_x)
    set_y = np.array(wasserstein_score_y)
    dist_vector = np.concatenate((set_x, set_y))
    mj1_wasserstein_distance = ((np.mean(set_x) + np.mean(set_y)) / 2)  # Reference 17
    print(mj1_wasserstein_distance)

    return mj1_wasserstein_distance

def changepoint_probabilities(df):
    n_segments = df['n_segments'].iloc[0]
    map_segments = df.loc[df['n_segments'] == n_segments]
    cutpoints = []
    probabilities = []
    max_prob_cutpoint = []
    for i in range(1,n_segments):
        cutpoint_x = map_segments.loc[map_segments['segment'] == i]
        cutpoint_locations = cutpoint_x['cut_point']
        cutpoint_weights = cutpoint_x['probability']

        # Convert to arrays
        cutpoint_locations_array = np.array(cutpoint_locations)
        cutpoint_weights_array = np.array(cutpoint_weights)

        # Compute MAP cutpoint
        max_probability_index = np.argmax(cutpoint_weights_array)
        map_cutpoint = cutpoint_locations_array[max_probability_index]

        # Append up to lists
        cutpoints.append(cutpoint_locations_array.flatten())
        probabilities.append(cutpoint_weights_array.flatten())
        max_prob_cutpoint.append(map_cutpoint.flatten())

    return cutpoints, probabilities, max_prob_cutpoint

def dendrogram_plot_test(matrix, distance_measure, data_generation, labels):

    # Compute and plot dendrogram.
    plt.rcParams.update({'font.size': 11})
    fig = pylab.figure(figsize=(15,10))
    axdendro = fig.add_axes([0.09,0.1,0.2,0.8])
    condensed_matrix = squareform(matrix) #MAX CHANGE
    Y = sch.linkage(condensed_matrix, method='average') #MAX CHANGE
    Z = sch.dendrogram(Y, orientation='right', labels=labels, leaf_rotation=360, leaf_font_size=12)
    # axdendro.set_xticks([])
    # axdendro.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = matrix[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    #axmatrix.set_xticks([])
    #axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)
    plt.savefig(data_generation+distance_measure+"Dendrogram",dpi=600) #MAX CHANGE
    plt.show()

    # Display and save figure.
    fig.show()

def transitivity_test(grid, matrix, distance):
    triangle_distance = []
    omega = np.zeros((len(grid), len(grid), len(grid)))
    for i in range(len(grid)):
        for j in range(len(grid)):
            for k in range(len(grid)):
                if distance == "hausdorff":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "modified_hausdorff":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "modified_hausdorff_2":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "modified_hausdorff_3":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "mj05":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "mj1":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "mj2":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "wasserstein":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
                if distance == "mj_wasserstein":
                    ratio = matrix[i, k] / (matrix[i, j] + matrix[j, k])
                    omega[i, j, k] = np.nan_to_num(ratio)
                    triangle_distance.append(ratio)
    triangle_distance = np.nan_to_num(triangle_distance)
    distances = triangle_distance.flatten()
    fail_values_avg = np.nan_to_num(distances[np.where(distances > 1.01)])
    fail_percentage = np.nan_to_num(len(fail_values_avg)/len(omega.flatten()))
    return fail_percentage, fail_values_avg

def plot_3d_mj_wasserstein(grid, distance_matrix, data_generation, distance_measure):
    triangle_distance = []
    omega = np.zeros((len(grid), len(grid), len(grid)))
    for i in range(len(grid)):
        for j in range(len(grid)):
            for k in range(len(grid)):
                ratio = distance_matrix[i, k] / (distance_matrix[i, j] + distance_matrix[j, k])
                omega[i, j, k] = np.nan_to_num(ratio)
                triangle_distance.append(ratio)

    print(triangle_distance)
    if np.max(np.nan_to_num(triangle_distance)) > 1:
        print("Element fails triangle inequality")
    else:
        print("No elements fail triangle inequality")

    # Make this bigger to generate a dense grid.
    N = len(grid)

    # Create some random data.
    volume = np.random.rand(N, N, N)

    # Create the x, y, and z coordinate arrays.  We use
    # numpy's broadcasting to do all the hard work for us.
    # We could shorten this even more by using np.meshgrid.
    x = np.arange(omega.shape[0])[:, None, None]
    y = np.arange(omega.shape[1])[None, :, None]
    z = np.arange(omega.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    # Set custom colour scheme
    c = np.tile(omega.ravel()[:, None], [1, 3])
    my_color = []
    for i in range(len(omega.ravel())):
        if omega.ravel()[i] <= 1.01:
            my_color.append('blue')
        if 1.01 < omega.ravel()[i] <= 2:
            my_color.append('yellow')
        if omega.ravel()[i] > 2:
            my_color.append('red')
    my_color = np.array(my_color)

    triangle_distance = np.nan_to_num(triangle_distance)
    distances = triangle_distance.flatten()
    fail_values_avg = np.nan_to_num(distances[np.where(distances > 1)])
    fail_percentage = np.nan_to_num(len(fail_values_avg)/len(omega.flatten()))

    print("fail values average", np.mean(fail_values_avg))
    print("fail percentage", fail_percentage)

    # my_color = np.where(omega.ravel() <= 1, 'blue', (np.where(1 < omega.ravel() < 2, 'yellow', 'red')))
    # col = np.where(x<1,'k',np.where(y<5,'b','r'))

    # col = np.where(x<1,'k',np.where(y<5,'b','r'))
    # Do the plotting in a single call.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x.ravel(),
               y.ravel(),
               z.ravel(),
               c=my_color)
    plt.xlabel('Country index')
    tick_loc = np.linspace(0,len(grid)-1,3)
    ax.set_xticks(tick_loc)
    ax.set_yticks(tick_loc)
    ax.set_zticks(tick_loc)
    plt.savefig(distance_measure+data_generation+"Transitivity")
    plt.show()

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

def chebyshev_distance(v1, v2):
    return np.max(np.abs(v1 - v2))

def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))

def cosine_distance(v1, v2):
    return np.matmul(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))