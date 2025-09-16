# We assume that PyTorch is already installed
import argparse
import json
import os
import pickle  # kept since original imports included it (even if unused)
import random
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader  # kept to avoid altering behavior/availability

from Utils import io as Utils


# ----------------------------
# Device & Reproducibility
# ----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print("Memory Allocated:", torch.cuda.memory_allocated(device) / (1024 ** 2), "MB")
    print("Memory Cached:", torch.cuda.memory_reserved(device) / (1024 ** 2), "MB")

torchversion = torch.__version__  # preserved from original


def seed_everything(seed: int = 42) -> None:
    """Set seeds for full reproducibility (same as original settings)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


# ----------------------------
# Core Execution
# ----------------------------
def execute(cfg: SimpleNamespace, pfe: str) -> None:
    """Main training/IO routine (logic unchanged)."""
    print("Loading parameters!")
    # Loading all parameter
    root_dir = cfg.root_dir
    models_dir = cfg.models_dir
    model_name = cfg.model_type
    model_id = cfg.model_id
    processed_dir = cfg.dataset_dir
    output_root_dir = cfg.output_root_dir
    output_features_dir = cfg.output_features_dir  # kept (unused) to avoid behavior change
    features_embedding_file_name = cfg.features_embedding_file_name  # kept (unused)
    output_log_dir = cfg.output_log_dir
    output_log_filename = cfg.output_log_filename

    path_dataset = os.path.join(root_dir, processed_dir, cfg.dataset_filename)
    path_adj_matrix = os.path.join(root_dir, processed_dir, cfg.adj_matrix_filename)

    # Graph weighting / topology parameter
    k = cfg.k

    # Training settings
    train_test_ratio = cfg.patients_train_test_ratio
    batch_training = cfg.batch_training
    batch_size = cfg.batch_size
    batch_log_size = cfg.batch_log_size
    batch_model_save_number = cfg.batch_model_save_number

    alpha = cfg.alpha  # kept (unused) for parity
    model_type = cfg.model_type
    hidden_dim = cfg.model_params.hidden_dim
    output_embedding_dim = cfg.model_params.output_embedding_dim  # embedding size
    num_hidden_layers = cfg.model_params.num_hidden_layers  # at least 2
    dropout_prob = cfg.model_params.dropout_prob
    epochs = cfg.model_params.epochs
    learning_rate = cfg.model_params.learning_rate
    weight_decay = cfg.model_params.weight_decay
    optimizer_type = cfg.model_params.optimizer_type  # 'Adam' or 'Sgd'
    best_model_val_loss = None
    best_model_f1 = None
    gamma = cfg.gamma

    if model_type == "GAT":
        heads = cfg.model_params.heads
    else:
        heads = 4

    if model_type == "Magnet":
        # K (int, optional): Order of the Chebyshev polynomial. Default: 2.
        # q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        q = cfg.model_params.q
        K = cfg.model_params.K
        Magnet_activation = cfg.model_params.Magnet_activation
    else:
        q = 0.25
        K = 2
        Magnet_activation = True

    print(model_type, "batch:", batch_training)

    # ----------------------------
    # Data Loading & Splitting
    # ----------------------------
    print("Loading data!")
    features, label, edge_index, edge_weight, raw_df = Utils.load_data(
        path_dataset, path_adj_matrix, k
    )
    PATID_list = raw_df["PATID"]

    # Get train/val/test subgraph data
    print("Spliting data!")
    train_data, val_data, test_data = Utils.split_data_by_PATID(
        PATID_list, train_test_ratio, features, label, edge_index, edge_weight
    )

    # Build whole-graph data (unchanged)
    data = Data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x=torch.tensor(features.values).type(torch.float),
        y=torch.tensor(label.values),
    )
    data.num_nodes = len(features)
    data.num_classes = len(label.unique())

    # Hyperparameters derived from data
    input_dim = data.num_features
    output_dim = data.num_classes

    # ----------------------------
    # Model Build & Optimizer
    # ----------------------------
    print("Building model!")
    model = Utils.build_model(
        model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_embedding_dim=output_embedding_dim,
        output_dim=output_dim,
        num_layers=num_hidden_layers,
        dropout_prob=dropout_prob,
        heads=heads,
        batch_training=batch_training,
        q=q,
        K=K,
        Magnet_activation=Magnet_activation,
    )

    # Move model to device (unchanged)
    model = model.to(device)

    # Optimizer
    optimizer = Utils.get_optimier(
        optimizer_type=optimizer_type,
        model_parameters=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ----------------------------
    # Training
    # ----------------------------
    print("Training model!")
    start_time = time.time()
    start_datetime = datetime.fromtimestamp(start_time)
    print("Start time: ", start_datetime.strftime("%y-%m-%d-%H:%M:%S"))

    trainning_log_df, best_model_val_loss, best_model_f1 = model.fit(
        train_data,
        val_data,
        epochs,
        optimizer,
        batch_size,
        device,
        batch_log_size,
        batch_model_save_number,
        output_root_dir,
        models_dir,
        model_name,
        model_id,
        gamma,
    )

    end_time = time.time()
    end_datetime = datetime.fromtimestamp(end_time)
    print("End time: ", end_datetime.strftime("%y-%m-%d-%H:%M:%S"))

    duration = end_time - start_time
    message = (
        f"The model {model_type} on dataset {model_id} took {duration/60:.2f} mins "
        f"to train {epochs} epochs."
    )
    print(message)

    # Save training time
    output_time_filename = "training_time.text"
    save_dir = os.path.join(output_root_dir, output_log_dir, model_name, model_id)
    save_path = Utils.check_saving_path(save_dir, output_time_filename)
    with open(save_path, "w") as f:
        f.write(message)

    # Save training log
    Utils.save_dataframe(
        trainning_log_df, output_root_dir, output_log_dir, model_name, model_id, output_log_filename
    )

    # Save best checkpoints (logic unchanged)
    if best_model_val_loss is not None:
        model.load_state_dict(best_model_val_loss)
        Utils.save_model(model, output_root_dir, models_dir, model_name, model_id + "best_val_loss")

    if best_model_f1 is not None:
        model.load_state_dict(best_model_f1)
        Utils.save_model(model, output_root_dir, models_dir, model_name, model_id + "best_f1")


# ----------------------------
# CLI Entrypoint
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", "-s", type=str, required=True)
    parser.add_argument("-p", "--profile", default="ipy_profile", help="Name of IPython profile to use")
    args = parser.parse_args()

    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))

    execute(cfg, args.profile)
