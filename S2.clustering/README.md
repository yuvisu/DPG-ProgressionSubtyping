# S2. Clustering Analysis

This folder contains code for clustering analysis of patient data, primarily using time series clustering methods to group patients based on their disease progression patterns.

## File Structure

```
S2.clustering/
├── 2.1.k_searching.py          # K-value search script
├── 2.2.clustering.py           # Main clustering script
├── Models/                     # Clustering model definitions
├── Output/                     # Output results directory
├── Settings/                   # Configuration files
│   ├── Clustering/
│   └── k3.json
└── Utils/                      # Utility functions
```

## Running Steps

### 1. K-value Search (2.1.k_searching.py)

This script is used to find the optimal number of clusters K by evaluating clustering performance using the Davies-Bouldin index.

**Run command:**
```bash
python 2.1.k_searching.py --setting Settings/k3.json
```

**Output files:**
- `search_magnet_k={k}_ninit_dbi.csv`: Contains Davies-Bouldin scores for different parameter combinations

### 2. Main Clustering Analysis (2.2.clustering.py)

This script performs the actual clustering analysis, dividing patients into different subtypes.

**Run command:**
```bash
python 2.2.clustering.py --setting Settings/Clustering/clustering_Magnet3.json
```

**Output files:**
- `Output/Cluster/{model_type}/{model_id}/Cluster_k{n_clusters}.csv`: Clustering results for complete dataset
- `Output/Cluster/{model_type}/{model_id}/Cluster_before_MCI_k{n_clusters}.csv`: Clustering results for pre-MCI dataset
- `Output/Cluster/{model_type}/{model_id}/TimeSeriesKMeans_k{n_clusters}.joblib`: Trained clustering model

## Data Requirements

Ensure the following data files exist:
- `../temp/nebor_200/output_node_features.pickle`: Patient feature data
- `../temp/nebor_200/adj_matrix.pickle`: Adjacency matrix
- `../temp/nebor_200_Results_gamma_0_dim_64/Output/feature_embeddings/{model_type}/whole_dataset/features_embedding.npy`: Feature embedding file
