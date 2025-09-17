# We assume that PyTorch is already installed
import torch
torchversion = torch.__version__
import pandas as pd
from torch_geometric.data import Data
import numpy as np
import os
import pickle
from Utils import io as Utils
import argparse
import json
from types import SimpleNamespace
from Models import GCN, GAT, GraphSAGE
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import joblib
from tslearn.utils import to_time_series_dataset
import joblib
from tslearn.clustering import silhouette_score
from sklearn.metrics import davies_bouldin_score
from tslearn.metrics import cdist_dtw

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def execute(cfg, pfe):
    # loading all parameter
    root_dir = cfg.root_dir
    model_name = cfg.model_type
    model_id = cfg.model_id
    processed_dir = cfg.dataset_dir
    output_root_dir = cfg.output_root_dir
    output_features_dir = cfg.output_features_dir
    features_embedding_file_name = cfg.features_embedding_file_name
    output_cluster_dir = cfg.output_cluster_dir
    cluster_filename = cfg.cluster_filename

    path_dataset = os.path.join(root_dir, processed_dir, cfg.dataset_filename)
    path_adj_matrix = os.path.join(root_dir, processed_dir,cfg.adj_matrix_filename)

    #cluster settings
    n_clusters = cfg.n_clusters
    distance_matrix = cfg.distance_matrix
    linkage = cfg.linkage
    
    path_dataset_before_MCI = os.path.join(root_dir, processed_dir+'_before_MCI', cfg.dataset_filename)
    path_adj_matrix_before_MCI  = os.path.join(root_dir, processed_dir+'_before_MCI',cfg.adj_matrix_filename)


    np.random.seed(42)
    # loading data
    _, _, _, _, raw_df = Utils.load_data(path_dataset, path_adj_matrix, 1)
    temp_df  = raw_df[['PATID', 'ENCID', 'ADMIT_DATE']]
                       
    _, _, _, _, raw_df_before_MCI = Utils.load_data(path_dataset_before_MCI, path_adj_matrix_before_MCI, 1)
    temp_df_before_MCI  = raw_df_before_MCI[['PATID', 'ENCID', 'ADMIT_DATE']]



    # loading learned embedding features
    embedding_feature = Utils.load_numpy(output_root_dir, output_features_dir, model_name, model_id, features_embedding_file_name)
    print('embedding_feature shapes : ',embedding_feature.shape)
    embedding_feature_length = embedding_feature.shape[1]
                       
    embedding_feature_before_MCI = Utils.load_numpy(output_root_dir, output_features_dir+'_before_MCI', model_name, model_id, features_embedding_file_name)
    print('embedding_feature _before_MCI shapes : ',embedding_feature_before_MCI.shape)
    embedding_feature_length_before_MCI = embedding_feature_before_MCI.shape[1]

    # Assuming the new features are named 'Feature1' and 'Feature2'
    feature_df = pd.DataFrame(embedding_feature, columns=['Feature'+str(i) for i in range(embedding_feature_length)])
    feature_df_before_MCI = pd.DataFrame(embedding_feature_before_MCI, columns=['Feature'+str(i) for i in range(embedding_feature_length_before_MCI)])
    # Concatenate temp_df with the feature embedding DataFrame
    temp_df = pd.concat([temp_df, feature_df], axis=1)
    temp_df_before_MCI = pd.concat([temp_df_before_MCI, feature_df_before_MCI], axis=1)






    print(temp_df)
    print()
    print(temp_df_before_MCI)

    # Preprocess train set
    temp_df_train = temp_df
    temp_df_train_before_MCI = temp_df_before_MCI
                       
    temp_df_train['ADMIT_DATE'] = pd.to_datetime(temp_df_train['ADMIT_DATE'])
    temp_df_train_before_MCI['ADMIT_DATE'] = pd.to_datetime(temp_df_train_before_MCI['ADMIT_DATE'])   
                       
    temp_df_train_sorted_df = temp_df_train.sort_values(by=['PATID', 'ADMIT_DATE'])
    temp_df_train_sorted_df_before_MCI = temp_df_train_before_MCI.sort_values(by=['PATID', 'ADMIT_DATE'])
                       
    # Group by 'PATID' and aggregate each group's features into a numpy array
    grouped_features = temp_df_train_sorted_df.groupby('PATID').apply(lambda group: group[['Feature' + str(i) for i in range(embedding_feature_length)]].values)
    sequences = [group for group in grouped_features]
    X = to_time_series_dataset(np.array(sequences, dtype="object"))
    print('Train set shape:', X.shape)
    np.random.seed(42)
    resampled_indices = np.random.choice(range(X.shape[0]), size=int(X.shape[0]), replace=False)
    X = X[resampled_indices, :]
    print(f"Shape of sampled data: {X.shape}")
                       
    # Group by 'PATID' and aggregate each group's features into a numpy array
    grouped_features_before_MCI = temp_df_train_sorted_df_before_MCI.groupby('PATID').apply(lambda group: group[['Feature' + str(i) for i in range(embedding_feature_length_before_MCI)]].values)
    sequences_before_MCI = [group for group in grouped_features_before_MCI]
    X_before_MCI = to_time_series_dataset(np.array(sequences_before_MCI, dtype="object"))
    print('Test set shape:', X_before_MCI.shape)

    # Train the clustering
    

    cluster_params = vars(cfg.cluster_params)
    print(cluster_params)

    Cluster = TimeSeriesKMeans(**cluster_params).fit(X)
    row1= [[n_clusters, cluster_params,Cluster.inertia_]]
    os.makedirs(os.path.join('Output', output_cluster_dir, model_name, model_id), exist_ok=True)
    outt = os.path.join('Output', output_cluster_dir, model_name, model_id, cluster_filename[:-4]+f'_elbow_score_k{n_clusters}.csv')
    pd.DataFrame(row1).to_csv(outt, index=False)  
    
    labels = Cluster.fit_predict(X)
    score = silhouette_score(X, labels)
    row = [[cluster_params, score]]
    os.makedirs(os.path.join('Output', output_cluster_dir, model_name, model_id), exist_ok=True)
    outt = os.path.join('Output', output_cluster_dir, model_name, model_id, cluster_filename[:-4]+f'_score_k{n_clusters}.csv')
    pd.DataFrame(row).to_csv(outt, index=False)
    print(cluster_params, score)
    
    # save clusert
    save_dir = os.path.join('Output', output_cluster_dir, model_name, model_id)
    cluster_name = f'TimeSeriesKMeans_k{n_clusters}.joblib'
    save_path = Utils.check_saving_path(save_dir, cluster_name)
    # Save the clustering model
    joblib.dump(Cluster, save_path)
    print(f"Model saved at {save_path}")

    # Preprocess the output set
    temp_sorted_df = temp_df.sort_values(by=['PATID', 'ADMIT_DATE'])

    # Group by 'PATID' and aggregate each group's features into numpy arrays
    grouped_features_with_indices = temp_sorted_df.groupby('PATID').apply(
        lambda group: (
            group[['Feature' + str(i) for i in range(embedding_feature_length)]].values,
            group.index.values
        )
    )
    # Extract sequences and PATIDs
    sequences = [group[0] for group in grouped_features_with_indices]
    sequence_indices = [temp_sorted_df.loc[group[1], 'PATID'].values[0] for group in grouped_features_with_indices]

    # Convert to time-series dataset and predict clusters
    Xtest = to_time_series_dataset(np.array(sequences))
    print('Output set shape:', Xtest.shape)

    labels = Cluster.predict(Xtest)

    # Assign cluster labels to the original dataframe
    patid_to_label = dict(zip(sequence_indices, labels))
    temp_df['cluster_info'] = temp_df['PATID'].map(patid_to_label)









          
                       
    # Preprocess the output set
    temp_sorted_df_before_MCI = temp_df_before_MCI.sort_values(by=['PATID', 'ADMIT_DATE'])

    # Group by 'PATID' and aggregate each group's features into numpy arrays
    grouped_features_with_indices_before_MCI = temp_sorted_df_before_MCI.groupby('PATID').apply(
        lambda group: (
            group[['Feature' + str(i) for i in range(embedding_feature_length_before_MCI)]].values,
            group.index.values
        )
    )
    # Extract sequences and PATIDs
    sequences_before_MCI = [group[0] for group in grouped_features_with_indices_before_MCI]
    sequence_indices_before_MCI = [temp_sorted_df_before_MCI.loc[group[1], 'PATID'].values[0] for group in grouped_features_with_indices_before_MCI]

    # Convert to time-series dataset and predict clusters
    Xtest_before_MCI = to_time_series_dataset(np.array(sequences_before_MCI))
    print('Output set shape:', Xtest_before_MCI.shape)

    labels_before_MCI = Cluster.predict(Xtest_before_MCI)

    # Assign cluster labels to the original dataframe
    patid_to_label_before_MCI = dict(zip(sequence_indices_before_MCI, labels_before_MCI))

    temp_df_before_MCI['cluster_info'] = temp_df_before_MCI['PATID'].map(patid_to_label_before_MCI)
                       
    
    temp_df = temp_df[['PATID','ENCID','ADMIT_DATE','cluster_info']]
    temp_df_before_MCI = temp_df_before_MCI[['PATID','ENCID','ADMIT_DATE','cluster_info']]

    print(temp_df)
    print(temp_df['cluster_info'].value_counts())
                       
    print(temp_df_before_MCI)
    print(temp_df_before_MCI['cluster_info'].value_counts())

    # save result to file
    #print(output_root_dir, output_cluster_dir, model_name, model_id, cluster_filename[:-4]+f'_k{n_clusters}.csv')
    Utils.save_dataframe(temp_df, 'Output', output_cluster_dir, model_name, model_id, cluster_filename[:-4]+f'_k{n_clusters}.csv')
    Utils.save_dataframe(temp_df_before_MCI, 'Output', output_cluster_dir, model_name, model_id, cluster_filename[:-4]+f'_before_MCI_k{n_clusters}.csv')
    






if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", "-s", type=str, required=True)

    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")

    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))


    execute(cfg, args.profile)
