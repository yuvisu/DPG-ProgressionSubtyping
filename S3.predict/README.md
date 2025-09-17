# S3. Prediction Modeling

This folder contains code for building prediction models based on clustering results from S2 stage, using machine learning methods to predict patient disease progression subtypes.

## File Structure

```
S3.predict/
├── 3.1.feature_processing.py   # Feature processing script
├── 3.2.model_predict.py        # Prediction model training script
├── Predict_model/              # Trained model storage directory
│   └── Magnet k=3/
└── shap_result/                # SHAP interpretability analysis results
    └── Magnet k=3/
```

## Running Steps

### 1. Feature Processing (3.1.feature_processing.py)

This script processes raw feature data and merges it with clustering results from S2 stage.

**Run command:**
```bash
python 3.1.feature_processing.py
```

**Output files:**
- `{model} k={k} data.csv`: Processed feature data with clustering labels

### 2. Prediction Model Training (3.2.model_predict.py)

This script trains multiple machine learning models to predict patient disease progression subtypes.

**Run command:**
```bash
python 3.2.model_predict.py
```

**Output files:**
- `Predict_model/{model} k={k}/model_{1-6}_grid.joblib`: Trained model files
- `Predict_model/{model} k={k}/macro_roc_curves.png`: ROC curve plots
- `shap_result/{model} k={k}/macro_roc_curves.png`: SHAP analysis ROC curve plots

## Data Requirements

Ensure the following data files exist:
- `before MCI features.pickle`: Raw feature data
- `../S2.clustering/Output/Cluster/{model}/whole_dataset/Cluster_k{k}.csv`: Complete dataset clustering results
- `../S2.clustering/Output/Cluster/{model}/whole_dataset/Cluster_before_MCI_k{k}.csv`: Pre-MCI dataset clustering results
