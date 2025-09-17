import numpy as np
import pandas as pd
import pickle
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans 
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import MinMaxScaler

###############################################
# Copyright Žiga Sajovic, XLAB 2019           #
# Distributed under the MIT License           #
#                                             #
# github.com/ZigaSajovic/Consensus_Clustering #
#                                             #
###############################################

import numpy as np
from itertools import combinations
import bisect
import concurrent.futures
import joblib
from tslearn.clustering import silhouette_score
from sklearn.metrics import davies_bouldin_score
from tslearn.metrics import cdist_dtw
import argparse
import pandas as pd
from torch_geometric.data import Data
import numpy as np

import os
import pickle
import argparse
import json
from types import SimpleNamespace
import os
import pickle
import argparse
import json
from types import SimpleNamespace
import numpy as np
import time

from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from tslearn.utils import to_time_series_dataset

from sklearn.metrics import davies_bouldin_score
def generate_multivariate_time_series(n_samples=100, min_timestamps=30, max_timestamps=100, n_features=3):
    """Generate synthetic multivariate time-series data with varied lengths."""
    np.random.seed(42)
    time_series = []
    for _ in range(n_samples):
        length = np.random.randint(min_timestamps, max_timestamps)
        series = np.random.rand(length, n_features)  # Shape: (timestamps, features)
        time_series.append(series)
    return time_series
def cluster_time_series(X, params):
    """Cluster time-series data using K-Means"""
    X_padded = to_time_series_dataset(X)  # Convert to tslearn format
    kmeans = TimeSeriesKMeans(**params)
    labels = kmeans.fit_predict(X_padded)
    centroids = kmeans.cluster_centers_  # Get cluster centroids
    return labels, centroids, X_padded
def compute_davies_bouldin_score(X, labels, centroids, metric="dtw"):
    """Custom Davies-Bouldin Score for time-series clustering"""

    n_clusters = len(centroids)
    cluster_dispersion = np.zeros(n_clusters)
    cluster_distances = np.zeros((n_clusters, n_clusters))

    # Compute intra-cluster dispersion (average distance to centroid)
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            dists = [dtw(x, centroids[i]) if metric == "dtw" else np.linalg.norm(x - centroids[i]) for x in cluster_points]
            cluster_dispersion[i] = np.mean(dists)

    # Compute inter-cluster separation (distance between centroids)
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                cluster_distances[i, j] = dtw(centroids[i], centroids[j]) if metric == "dtw" else np.linalg.norm(centroids[i] - centroids[j])

    # Compute Davies-Bouldin Index
    db_indexes = []
    for i in range(n_clusters):
        ratios = [(cluster_dispersion[i] + cluster_dispersion[j]) / cluster_distances[i, j] for j in range(n_clusters) if i != j]
        db_indexes.append(max(ratios))  # Maximum ratio for each cluster

    return np.mean(db_indexes)  # Final Davies-Bouldin Score
# Compute Davies-Bouldin Score

def compute_davies_bouldin_score_sklearn(X_scaled, labels):
    """Compute Davies-Bouldin Score using sklearn's implementation."""
    
    
    X_transform = np.nan_to_num(X_scaled)
    
    
    X_flattened = X_transform.reshape(X_transform.shape[0], -1)  # Flatten time series for sklearn compatibility
    #print(X_scaled)
    
    return davies_bouldin_score(X_flattened, labels)


 
    
    
    
    
    
# Set the random seed for reproducibility
#np.random.seed(42)

def execute(cfg, pfe):
    cluster = cfg.n_clusters
    print('cluster', cluster)
            
    #loading orginal data
    path_dataset = "../temp/nebor_200/output_node_features.pickle"
    # loading files
    file_dataset = open(path_dataset, 'rb')
    raw_df = pickle.load(file_dataset)
    raw_df.reset_index(inplace=True)
    temp_df  = raw_df[['PATID', 'ENCID', 'ADMIT_DATE']]

    #Read the indices from the CSV
    #read_indices = pd.read_csv("../Utils/Sampling_index/train_indices.csv")["index"].tolist()
    root_path = "../temp/nebor_200_Results_gamma_0_dim_64/Output/feature_embeddings/"
    dataset = 'whole_dataset'
    file_name = "features_embedding.npy"
    model = "Magnet"


    # List of input files and their corresponding names
    files = {
        "GCN": root_path + 'GCN' + "/"+ dataset +"/"+ file_name,
        "GAT": root_path + 'GAT' + "/"+ dataset +"/"+ file_name,
        "GraphSAGE": root_path + 'GraphSAGE' + "/"+ dataset +"/"+ file_name,
        "Magnet": root_path + 'Magnet' + "/"+ dataset +"/"+ file_name
    }

    path = files[model]
    print(model, path)

    # Load the .npy file
    data = np.load(path)
    # Print the shape of the loaded data
    print(f"Shape of {model} data embedings: {data.shape}")
    # Use these indices to get the sampled rows from the data
    embedding_feature = data
    print(f"Shape of {model} Train set data embedings: {embedding_feature.shape}")

    # Assuming the new features are named 'Feature1' and 'Feature2'
    embedding_feature_length = embedding_feature.shape[1]
    feature_df = pd.DataFrame(embedding_feature, columns=['Feature'+str(i) for i in range(embedding_feature_length)])
    # Concatenate temp_df with the feature embedding DataFrame
    temp_df = pd.concat([temp_df, feature_df], axis=1)
    print(temp_df)

    # preprocess data 
    # Preprocess train set
    temp_df_train = temp_df#.iloc[read_indices, : ]


    temp_df_train_sorted_df = temp_df_train.sort_values(by=['PATID', 'ADMIT_DATE'])
    # Group by 'PATID' and aggregate each group's features into a numpy array
    grouped_features = temp_df_train_sorted_df.groupby('PATID').apply(lambda group: group[['Feature' + str(i) for i in range(embedding_feature_length)]].values)
    sequences = [group for group in grouped_features]
    X = to_time_series_dataset(np.array(sequences, dtype="object"))
    print('Train set shape after preprocess:', X.shape)

    np.random.seed(42)
    resampled_indices = np.random.choice(range(X.shape[0]), size=int(X.shape[0]), replace=False)
    X = X[resampled_indices, :]
    # Print the shape of the sampled data
    print(f"Shape of sampled {model} data: {X.shape}")


    # Define parameter grid
    param_grid = {
        "kmeans__n_clusters": [cluster],
        "metric": [ "dtw", "softdtw"],
        # "max_iter": [50, 100, 200, 400],
        "n_init": [5, 10, 15, 20],
        # "tol":[1e-6, 1e-5, 1e-7, 1e-8]
    }

    scores = []
    best_score = 0
    best_params = None
    best_model = None
    row = []
    # Iterate through parameter combinations
    for k in param_grid["kmeans__n_clusters"]:
        for n_init in param_grid["n_init"]:
            for metric in param_grid["metric"]:
                # for max_iter in param_grid["max_iter"]:
                    # for tol in param_grid["tol"]:
                        print(k)
                        # ts_kmeans = TimeSeriesKMeans(n_clusters=k, n_init=n_init, metric=metric, max_iter=max_iter, tol=tol, random_state=42)
                        params = {
                            'n_clusters':k,
                            'metric':metric, 
                             'n_init':n_init, 
                            'random_state':42
                        }
                        
                        
                        
                        start_time = time.time()

                        labels, centroids,X_padded = cluster_time_series(X, params)

                        end_time = time.time()

                        execution_time = end_time - start_time

                        print("K mean Execution time:", execution_time)

                        start_time = time.time()

                        db_score = compute_davies_bouldin_score(X_padded, labels, centroids)

                        end_time = time.time()

                        execution_time = end_time - start_time

                        print("Execution time:", execution_time)

                        print(f"Davies-Bouldin Score (Time-Series): {db_score:.4f}")

                        
                        

                        
                        print(f'params metric {metric}; n_init {n_init};')
                        row.append([k, metric, n_init, db_score])


    temp = pd.DataFrame(row, columns=['k', 'metric', 'n_init', 'db_score'])
    temp.to_csv(f'search_magnet_k={k}_ninit_dbi.csv', index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", "-s", type=str, required=True)

    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")

    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))


    execute(cfg, args.profile)