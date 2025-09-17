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
from torch_geometric.nn import GCNConv
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from Utils import io as Utils
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

#------------------------------------------DGCN layer---------------------------------------------------------
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class DGCNConv(MessagePassing):
    r"""An implementatino of the graph convolutional operator from the
    `Directed Graph Convolutional Network
    <https://arxiv.org/pdf/2004.13970.pdf>`_ paper.
    The same as Kipf's GCN but remove trainable weights.
    Args:
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True, **kwargs):

        #kwargs.setdefault('aggr', 'add')
        super(DGCNConv, self).__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """
        Making a forward pass of the graph convolutional operator.
        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (Adj) - Edge indices.
            * edge_weight (OptTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * out (PyTorch FloatTensor) - Hidden state tensor for all nodes.
        """
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

#------------------------------------------DGCN model class-----------------------------------------------------


class DGCN_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_embedding_dim, output_dim, num_layers, dropout_prob, improved=False, cached=False):
        super(DGCN_model, self).__init__()

        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.dgconv = DGCNConv(improved=improved, cached=cached)

        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        for i in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim * 3, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim * 3, output_embedding_dim, bias=False))

        self.Conv = nn.Conv1d(output_embedding_dim * 3,output_dim, kernel_size=1)

        self.bias = nn.ParameterList()
        for i in range(num_layers - 1):
            self.bias.append(nn.Parameter(torch.Tensor(1, hidden_dim)))
            nn.init.zeros_(self.bias[-1])
        self.bias.append(nn.Parameter(torch.Tensor(1, output_embedding_dim)))
        nn.init.zeros_(self.bias[-1])

    def forward(self, x, edge_index, edge_in, edge_out, in_w=None, out_w=None):
        for i in range(self.num_layers):
            x = self.linears[i](x)
            x1 = self.dgconv(x, edge_index)
            x2 = self.dgconv(x, edge_in, in_w)
            x3 = self.dgconv(x, edge_out, out_w)

            x1 += self.bias[i]
            x2 += self.bias[i]
            x3 += self.bias[i]

            x = torch.cat((x1, x2, x3), axis=-1)
            x = F.relu(x)

        if self.dropout_prob > 0:
            x = F.dropout(x, self.dropout_prob, training=self.training)

        y = x.unsqueeze(0)
        y = y.permute((0, 2, 1))
        y = self.Conv(y)
        y = y.permute((0, 2, 1)).squeeze()

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
        # calculatng the edges and weight
        train_edge_index, train_edge_in, train_in_weight, train_edge_out, train_out_weight = Utils.directed_features_in_out(
            training_data.edge_index, len(training_data.x), training_data.edge_weight)
        val_edge_index, val_edge_in, val_in_weight, val_edge_out, val_out_weight = Utils.directed_features_in_out(
            val_data.edge_index, len(val_data.x), val_data.edge_weight)


        for epoch in range(epochs + 1):
            # Training
            optimizer.zero_grad()
            embedding, out = self(training_data.x, train_edge_index,
                                    train_edge_in, train_edge_out, train_in_weight, train_out_weight)
            loss = criterion(out, training_data.y)
            acc = Utils.accuracy(out.argmax(dim=1), training_data.y)
            loss.backward()
            optimizer.step()

            # Validation
            with torch.no_grad():
                embedding_val, out_val = self(val_data.x, val_edge_index,
                                              val_edge_in,  val_edge_out, val_in_weight, val_out_weight)
                val_loss = criterion(out_val, val_data.y)
                val_acc = Utils.accuracy(out_val.argmax(dim=1), val_data.y)

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



if __name__ == "__main__":
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


    # Building whole graph data
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
    hidden_dim = 16
    output_embedding_dim = 16 # embedding size
    output_dim = data.num_classes
    num_GCN_layers = 2
    dropout_prob = 0.7
    epochs = 100
    learning_rate = 0.01
    weight_decay = 5e-4
    optimizer = 'Adam'

    DGCNmodel = DGCN_model(input_dim = input_dim, hidden_dim = hidden_dim,output_embedding_dim = output_embedding_dim,  output_dim = output_dim,
                          num_layers = num_GCN_layers, dropout_prob = dropout_prob)
    DGCNmodel.to(device)


    # Define the optimizers to use
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(DGCNmodel.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'Sgd':
        optimizer = torch.optim.SGD(DGCNmodel.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(DGCNmodel.parameters(), lr=learning_rate, weight_decay=weight_decay)

    DGCNmodel.fit(train_data, val_data, epochs=epochs, optimizer=optimizer)

    edge_index, edge_in, in_weight, edge_out, out_weight = Utils.directed_features_in_out(
        data.edge_index, len(data.x), data.edge_weight)


    output_from_model = DGCNmodel(data.x, edge_index, edge_in, edge_out, in_weight, out_weight)
    embedding_feature = output_from_model[0].cpu().detach().numpy()
    print(embedding_feature.shape)
    print(Utils.accuracy(output_from_model[1].argmax(dim=1),data.y))
    """