# We assume that PyTorch is already installed
import argparse
import json
import os
import pickle  # kept from original imports
import random
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader  # kept from original imports

from Utils import io as Utils

torchversion = torch.__version__  # preserved

# ----------------------------
# Device & CUDA info
# ----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print("Memory Allocated:", torch.cuda.memory_allocated(device) / (1024 ** 2), "MB")
    print("Memory Cached:", torch.cuda.memory_reserved(device) / (1024 ** 2), "MB")


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ----------------------------
# Main execution (logic unchanged)
# ----------------------------
def execute(cfg, pfe):
    print("Loading parameters!")
    # loading all parameter
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    model_name = cfg.model_type
    model_id = cfg.model_id
    processed_dir = cfg.dataset_dir
    output_root_dir = cfg.output_root_dir
    output_features_dir = cfg.output_features_dir
    features_embedding_file_name = cfg.features_embedding_file_name
    output_log_dir = cfg.output_log_dir
    output_log_filename = cfg.output_log_filename

    path_dataset = os.path.join(root_dir, processed_dir, cfg.dataset_filename)
    path_adj_matrix = os.path.join(root_dir, processed_dir, cfg.adj_matrix_filename)

    # weight degree rev_sigmoid
    k = cfg.k

    # training settings
    train_test_ratio = cfg.patients_train_test_ratio
    batch_training = cfg.batch_training
    batch_size = cfg.batch_size
    batch_log_size = cfg.batch_log_size
    batch_model_save_number = cfg.batch_model_save_number

    alpha = cfg.alpha
    model_type = cfg.model_type  # could be 'GAT', 'GraphSAGE', 'DIGCN', 'DGCN'
    hidden_dim = cfg.model_params.hidden_dim
    output_embedding_dim = cfg.model_params.output_embedding_dim  # embedding size
    num_hidden_layers = cfg.model_params.num_hidden_layers  # at least 2
    dropout_prob = cfg.model_params.dropout_prob
    epochs = cfg.model_params.epochs
    learning_rate = cfg.model_params.learning_rate
    weight_decay = cfg.model_params.weight_decay
    optimizer_type = cfg.model_params.optimizer_type  # could be 'Adam' or 'Sgd' else will be 'Adam'
    best_model_val_loss = None
    best_model_f1 = None
    gamma = cfg.gamma

    if model_type == "GAT":
        heads = cfg.model_params.heads
    else:
        heads = 4

    if model_type == "Magnet":
        # K (int, optional): Order of the Chebyshev polynomial.  Default: 2.
        # q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        q = cfg.model_params.q
        K = cfg.model_params.K
        Magnet_activation = cfg.model_params.Magnet_activation
    else:
        q = 0.25
        K = 2
        Magnet_activation = True

    print(model_type, "batch:", batch_training)

    # loading data
    print("Loading data!")
    features, label, edge_index, edge_weight, raw_df = Utils.load_data(
        path_dataset, path_adj_matrix, k
    )
    PATID_list = raw_df["PATID"]

    # Building whole graph data
    data = Data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x=torch.tensor(features.values).type(torch.float),
        y=torch.tensor(label.values),
    )
    data.num_nodes = len(features)
    data.num_classes = len(label.unique())

    # loading hyperparameters
    input_dim = data.num_features
    output_dim = data.num_classes

    # build model
    print("loading model!")

    # continue load previous model
    model = Utils.load_model(output_root_dir, models_dir, model_name, model_id + "best_f1")

    # loading model and data to GPU
    model = model.to(device)

    # get feature embedding from best model
    print("Get feature embedding from best model!")

    # for batching tranining
    if model_type == "Magnet":
        embedding_feature = Utils.embedding_batch_Magnet(model, data, batch_size, device)
    elif model_type == "GCN":
        embedding_feature = Utils.embedding_batch_GCN(model, data, batch_size, device)
    else:
        embedding_feature = Utils.embedding_batch(model, data, batch_size, device)

    print(output_root_dir, output_features_dir, model_name, model_id, features_embedding_file_name)
    # save the embedding_features
    Utils.save_numpy(
        embedding_feature,
        output_root_dir,
        output_features_dir,
        model_name,
        model_id,
        features_embedding_file_name,
    )

    print(embedding_feature.shape)


# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, required=True)
    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")
    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))

    execute(cfg, args.profile)
