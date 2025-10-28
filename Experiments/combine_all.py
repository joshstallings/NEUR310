
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from netrep.metrics import LinearMetric, GaussianStochasticMetric

np.random.seed(42)

# Array of paths (as strings) of the data I want to analyze
### Load in the data ###
DATA = [
    # 3M Model data
    "./data/3M/3M_experiment800N_9_TEMPS_PROMPTA_per_trial_tensor.npy",
    "./data/3M/3M_experiment800N_9_TEMPS_PROMPTB_per_trial_tensor.npy",
    # 8M Model data
    "./data/8M/8M_experiment800N_9_TEMPS_PROMPTA_per_trial_tensor.npy",
    "./data/8M/8M_experiment800N_9_TEMPS_PROMPTB_per_trial_tensor.npy"
]

NUM_TEMPS = 9
NUM_SEEDS = 3
NUM_EXPERIMENTS = 800
NUM_TOKENS = 23 # (num_tokens_to_generate + prompt_length - 1)
N_COMPONENTS = 10

def cov3d(data):
    '''
    Calculates the covariance for a 3D array.
    Iterates over L (some number of parameters or samples) and calculates the covariance.
    '''
    if(len(data.shape) != 3):
        print("ERROR NOT 3D")
        return None

    L, _, M = data.shape # (num_tokens, num_experiments, num_neurons)
    covar_matrices = np.zeros((L, M, M))
    for i in range(L):
        covar_matrices[i] = np.cov(data[i, :, :], rowvar=False)
    
    return covar_matrices

def main():
    ### Load data Apply PCA ###
    pca = PCA(n_components=N_COMPONENTS)
    data_matrix = np.empty((len(DATA), NUM_TEMPS, NUM_SEEDS, NUM_EXPERIMENTS, NUM_TOKENS, N_COMPONENTS ))
    for i in range(len(DATA)):
        print(f"{i+1}/{len(DATA)}")
        per_trial_data = np.load(DATA[i])
        per_trial_data_pca = np.empty((NUM_TEMPS, NUM_SEEDS, NUM_EXPERIMENTS, NUM_TOKENS, N_COMPONENTS))
        # Loop through the num_temps, num_seeds, num_experiments to reduce each trial to its
        # principle 10 components
        for j in range(per_trial_data_pca.shape[0]):
            for k in range(per_trial_data_pca.shape[1]):
                for n in range(NUM_EXPERIMENTS):
                    per_trial_data_pca[j, k, n, :, :] = pca.fit_transform(per_trial_data[j, k, n, :, :])
        
        data_matrix[i] = per_trial_data_pca


    ### Caluclate the mean trajectories ### 
    mean_data_matrix = np.mean(data_matrix, 3)
    mean_data_matrix = mean_data_matrix.reshape(mean_data_matrix.shape[0] * mean_data_matrix.shape[1] * mean_data_matrix.shape[2],
                                                mean_data_matrix.shape[3], mean_data_matrix.shape[4])


    ### Calculate the covariance of the trajectories #### 
    reshaped_for_covar = data_matrix.reshape(data_matrix.shape[0] * data_matrix.shape[1] * data_matrix.shape[2],
                                            data_matrix.shape[4], data_matrix.shape[3], data_matrix.shape[5])

    # input to metric is (23, 10) and cov needs to be (23, 10, 10)
    covars = []
    for i in range(reshaped_for_covar.shape[0]):
        covar = cov3d(reshaped_for_covar[i])
        covars.append(covar)


    ### LINEAR PROCRUSTES DISTANCE ###
    linear_distances = [] # array of distances

    # # Array of [ (Time points, Neurons), ... ]
    Xs = []
    for i in range(mean_data_matrix.shape[0]):
        Xs.append(mean_data_matrix[i])

    print(Xs[0].shape)
    alphas = [0, 0.5, 1] # alphas to use for linear metric

    for alpha in alphas:
        metric = LinearMetric(alpha=alpha)
        test_dist, _ = metric.pairwise_distances(Xs)
        linear_distances.append(test_dist)


    ### CALCULATES STOCHASTIC SHAPE DISTANCE ###
    gaussian_distances = []
    # Get covariance over N experiemnts and create input array
    Xs = []
    for i in range(mean_data_matrix.shape[0]):
        Xs.append((mean_data_matrix[i, :, :], covars[i]))

    alphas = [0, 1, 2] # 0 only uses covar, 1 = 2-Wasserstein, 2 = only uses means
    for alpha in alphas:
        metric = GaussianStochasticMetric(alpha=alpha, init='rand', n_restarts=50)
        test_dist, _ = metric.pairwise_distances(Xs)
        gaussian_distances.append(test_dist)


    ### Export distance matrices to system ### 
    for i in range(len(linear_distances)):
        d = linear_distances[i]
        np.save(f"./data/linear_distance_matrix_alpha_{i}", d)

    # for i in range(len(gaussian_distances)):
    #     d = gaussian_distances[i]
    #     np.save(f"./data/gaussian_distance_matrix_alpha_{i}", d)


if __name__ == "__main__":
    main()