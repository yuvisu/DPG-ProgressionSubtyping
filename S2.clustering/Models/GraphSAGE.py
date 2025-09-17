# We assume that PyTorch is already installed
import torch
torchversion = torch.__version__
import pandas as pd
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import Utils
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_embedding_dim, output_dim, num_layers, dropout_prob):
        super(GraphSAGE, self).__init__()

        self.num_layers = num_layers

        self.sages = torch.nn.ModuleList()
        self.sages.append(SAGEConv(input_dim, hidden_dim))
        for i in range(num_layers - 2):
            self.sages.append(SAGEConv(hidden_dim, hidden_dim))
        self.sages.append(SAGEConv(hidden_dim,output_embedding_dim))

        self.fc = nn.Linear(output_embedding_dim, output_dim)

        self.dropout_prob = dropout_prob


    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.sages[i](x=x, edge_index=edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.sages[self.num_layers - 1](x=x, edge_index=edge_index)
        y = self.fc(x)
        return x, F.log_softmax(y, dim=1)

    def fit(self,  training_data, val_data, epochs, optimizer):
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_model_state = None

        acc_list = []
        loss_list = []
        val_acc_list = []
        val_loss_list = []
        f1_list=[]
        auroc_list = []
        sensitivity_list = []
        specificity_list = []

        self.train()
        for epoch in range(epochs + 1):
            # Training
            optimizer.zero_grad()
            embedding, out = self(training_data.x, training_data.edge_index)
            loss = criterion(out, training_data.y)
            acc = accuracy(out.argmax(dim=1), training_data.y)
            loss.backward()
            optimizer.step()

            # Validation
            with torch.no_grad():
                embedding_val, out_val = self(val_data.x, val_data.edge_index)
                val_loss = criterion(out_val, val_data.y)
                val_acc = accuracy(out_val.argmax(dim=1), val_data.y)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.state_dict()

            # calculate score
            pred_test = out_val.argmax(dim=1).cpu().detach().numpy()
            true_test = val_data.y.cpu().detach().numpy()
            f1 = f1_score(true_test, pred_test, average='weighted')
            # precision = precision_score(true_test,pred_test, average='weighted', zero_division=1)
            # recall = recall_score(true_test,pred_test, average='weighted',zero_division=1)
            # cm = confusion_matrix(true_test,pred_test)

            # AUROC
            num_classes = out_val.shape[1]
            auroc_scores = []
            for i in range(num_classes):
                try:
                    auroc = roc_auc_score(true_test == i, out_val[:, i].cpu().detach().numpy())
                except ValueError:
                    auroc = 0.0
                auroc_scores.append(auroc)
            auroc = np.mean(auroc_scores)

            # Sensitivity and Specificity
            sensitivity_scores = []
            specificity_scores = []
            for i in range(num_classes):
                true_class = true_test == i
                pred_class = pred_test == i

                tp = np.sum(true_class & pred_class)
                tn = np.sum(~true_class & ~pred_class)
                fp = np.sum(~true_class & pred_class)
                fn = np.sum(true_class & ~pred_class)

                sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

                sensitivity_scores.append(sensitivity)
                specificity_scores.append(specificity)

            sensitivity = np.mean(sensitivity_scores)
            specificity = np.mean(specificity_scores)

            # Print metrics every epoch
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc * 100:>6.2f}% '
                  f'| Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}% '
                  f'| F1 score: {f1:.3f} | AUROC: {auroc:.3f} '
                  f'| Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}')

            acc_list.append(acc)
            loss_list.append(loss.cpu().detach().numpy())
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss.cpu().detach().numpy())
            f1_list.append(f1)
            auroc_list.append(auroc)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)

        # create a dictionary with column names as keys and lists as values
        trainning_data = {'Epoch': range(epochs + 1), 'Train_acc': acc_list, 'Train_loss': loss_list,
                          'Val_acc': val_acc_list, 'Val_loss': val_loss_list, 'F1_val': f1_list, 
                          'Auroc':auroc_list, 'Sensitivity':sensitivity_list, 'Specificity':specificity_list}
        result_df = pd.DataFrame(trainning_data)

        # Load best model state
        self.load_state_dict(best_model_state)

        epochs_range = range(len(acc_list))
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc_list, label='Training Accuracy')
        plt.plot(epochs_range, val_acc_list, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss_list, label='Training Loss')
        plt.plot(epochs_range, val_loss_list, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        # plt.savefig('Graph_model_results/GCN_training.png')
        # plt.show()

        return result_df

#if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()                                               

    parser.add_argument("--setting", "-s", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.setting) as json_file:
        cfg = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))

    """
    """
    # loading files
    file = open('Graph_dataset/node_features.pickle', 'rb')
    df_features = pickle.load(file)
    file = open('Graph_dataset/adj_matrix.pickle', 'rb')
    adj_matrix = pickle.load(file)

    # fliter the features and reset index
    features = df_features.iloc[:, 4:]
    features.index = range(len(features))

    # Loading edge info convert to coo format
    coo = adj_matrix.tocoo()
    weighted_edge_data = pd.DataFrame({'source': coo.row, 'target': coo.col, 'weight': coo.data})
    # loading edge_weight and edge_index
    row = torch.from_numpy(coo.row.astype(np.int64)).type(torch.long)
    col = torch.from_numpy(coo.col.astype(np.int64)).type(torch.long)
    edge_weight = torch.from_numpy(coo.data.astype(np.float64)).type(torch.float)
    edge_index = torch.stack([row, col], dim=0)

    # loading the label from the dataset
    label = df_features['next_status'].astype(int)
    label.index = range(len(label))


    # Building graph data
    data = Data(edge_index=edge_index, edge_weight=edge_weight)
    data.num_nodes = len(features)
    # loading embedding
    data.x = torch.tensor(features.values).type(torch.float)
    # loading labels
    y = torch.tensor(label.values)
    data.y = y.clone().detach()
    data.num_classes = len(label.unique())

    # get unique PATID list
    PATID_list = df_features['PATID']
    train_data, val_data = Utils.split_data_by_PATID(PATID_list, 0.3, features, label, edge_index, edge_weight)

    # loading data to GPU or CPU
    data = data.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    #loading hyperparameters
    input_dim = data.num_features
    output_dim = data.num_classes
    hidden_dim = 16
    output_embedding_dim = 16# embedding size
    num_hidden_layers = 2
    dropout_prob = 0.7
    epochs = 100
    learning_rate = 0.01
    weight_decay = 5e-4
    optimizer = 'Adam'

    GraphSAGE = GraphSAGE(input_dim = input_dim, hidden_dim = hidden_dim, output_embedding_dim = output_embedding_dim,
                          output_dim = output_dim,num_layers = num_hidden_layers, dropout_prob = dropout_prob)
    GraphSAGE.to(device)

    # Define the optimizers to use
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(GraphSAGE.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'Sgd':
        optimizer = torch.optim.SGD(GraphSAGE.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(GraphSAGE.parameters(), lr=learning_rate, weight_decay=weight_decay)

     # Trainning
    GraphSAGE.fit(train_data, val_data, epochs=epochs, optimizer=optimizer)

    output_from_model = GraphSAGE(data.x, data.edge_index)
    embedding_feature = output_from_model[0].cpu().detach().numpy()
    print(embedding_feature.shape)
    """