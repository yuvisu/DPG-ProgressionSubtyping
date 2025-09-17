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
from torch_geometric.nn import GCNConv
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from Utils import io as Utils
#from torch_geometric.loader import NeighborSampler (old verison)
from torch_geometric.loader import NeighborLoader
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#------------------------------------------DIGCN layer---------------------------------------------------------
import torch
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros


class DiGCNConv(MessagePassing):
    r"""The graph convolutional operator from the
    `Digraph Inception Convolutional Networks
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the adj matrix on first execution, and will use the
            cached version for further executions.
            Please note that, all the normalized adj matrices (including undirected)
            are calculated in the dataset preprocessing to reduce time comsume.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: int, out_channels: int, improved: bool = False, cached: bool = False,
                 bias: bool = True, **kwargs):
        super(DiGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x: torch.FloatTensor, edge_index: torch.LongTensor,
                edge_weight: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        Making a forward pass of the DiGCN Convolution layer.
        Arg types:
            * x (PyTorch FloatTensor) - Node features.
            * edge_index (PyTorch LongTensor) - Edge indices.
            * edge_weight (PyTorch FloatTensor, optional) - Edge weights corresponding to edge indices.
        Return types:
            * x (PyTorch FloatTensor) - Hidden state tensor for all nodes.
        """
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None and edge_index.size(1) != self.cached_num_edges:
            raise RuntimeError(
                'Cached {} number of edges, but found {}. Please '
                'disable the caching behavior of this layer by removing '
                'the `cached=True` argument in its constructor.'.format(
                    self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please '
                    'obtain the adj matrix in preprocessing.')
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

#------------------------------------------DIGCN model class-----------------------------------------------------

class DIGCN_batch(torch.nn.Module):
    r"""An implementation of the DiGCN model without inception blocks for node classification from the
    `Digraph Inception Convolutional Networks"""
    def __init__(self, input_dim, hidden_dim, output_embedding_dim, output_dim, num_layers, dropout_prob):
        super(DIGCN_batch, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(DiGCNConv(input_dim, hidden_dim))
        for i in range(num_layers - 2):
            self.convs.append(DiGCNConv(hidden_dim, hidden_dim))
        self.convs.append(DiGCNConv(hidden_dim, output_embedding_dim))
        self.fc = nn.Linear(output_embedding_dim, output_dim)
        self.dropout_prob = dropout_prob

    def forward(self, x, edge_index, edge_weight):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight))
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.convs[self.num_layers - 1](x=x, edge_index=edge_index, edge_weight=edge_weight)
        y = self.fc(x)
        return x, F.log_softmax(y, dim=1)
    
    def fit(self,  training_data, val_data, epochs, optimizer, batch_size, device, alpha):
        criterion = torch.nn.CrossEntropyLoss()
        train_loader = NeighborLoader(training_data, num_neighbors=[-1]*self.num_layers, batch_size=batch_size, shuffle =True)
        val_loader = NeighborLoader(val_data, num_neighbors=[-1]*self.num_layers, batch_size=batch_size)
        print('number of batch in trainning data: ', len(train_loader))
        #preprocess
        print('start Train data')
        for batch in train_loader:
            batch.edge_index, batch.edge_weight = Utils.get_appr_directed_adj(alpha, batch.edge_index, batch.x.shape[0], batch.x.dtype, batch.edge_weight)
        print('start Val data')
        for batch in val_loader:
            batch.edge_index, batch.edge_weight = Utils.get_appr_directed_adj(alpha, batch.edge_index, batch.x.shape[0], batch.x.dtype, batch.edge_weight)


        best_val_loss = float('inf')
        best_f1_score = 0
        best_model_val_loss = None
        best_model_f1 = None
        
        acc_list = []
        loss_list = []
        val_acc_list = []
        val_loss_list = []
        f1_list=[]
        auroc_list = []
        precision_list = []
        recall_list = []
        sensitivity_list = []
        specificity_list = []
        cm_list = []
        index_list=[]
        index = 0
    
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_total_loss = 0.0
            train_total_correct = 0
            train_total_samples = 0
            for batch in train_loader:
                optimizer.zero_grad()
                # push every thing to device
                batch_size = batch.batch_size
                x_batch = batch.x.to(device)
                y_batch = batch.y.to(device)
                batch_edge_index = batch.edge_index.to(device)
                batch_edge_weight = batch.edge_weight.to(device)
            
                embedding, out = self(x_batch, batch_edge_index, batch_edge_weight)
                
                # only using batch node for calculate loss and back 
                train_loss = criterion(out[:batch_size], y_batch[:batch_size])
                train_loss.backward()
                optimizer.step()
                
                #  Compute loss for this batch
                train_total_samples += batch_size
                train_total_loss += float(train_loss) * batch_size
                # Compute accuracy for this batch
                preds = out[:batch_size].argmax(dim=1)
                #train_total_correct += (preds == y_batch[:batch_size]).sum().item()
                batch_total_correct = (preds == y_batch[:batch_size]).sum().item()

                # At the end of the loop:
                del x_batch, y_batch, batch_edge_index, batch_edge_weight
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Compute overall accuracy and loss
                #acc = train_total_correct / train_total_samples
                #loss = train_total_loss / train_total_samples
                # Compute batch accuracy and loss   
                acc = batch_total_correct/batch_size
                
                # calcuate log every 50 batch
                index+=1
                if index%50 == 1:
                    # Validation
                    self.eval() 
                    with torch.no_grad():
                        val_total_loss = 0.0
                        val_total_correct = 0
                        val_total_samples = 0
                        all_preds = []
                        all_labels = []
                        for batch in val_loader:
                            # push every thing to device
                            batch_size = batch.batch_size
                            x_batch = batch.x.to(device)
                            y_batch = batch.y.to(device)
                            batch_edge_index = batch.edge_index.to(device)
                            batch_edge_weight = batch.edge_weight.to(device)

                            embedding, out = self(x_batch, batch_edge_index, batch_edge_weight)

                            # only using batch node for calculate loss and back 
                            out = out[:batch_size]
                            loss = criterion(out, y_batch[:batch_size])

                            val_total_samples += batch_size
                            val_total_loss += float(loss) * batch_size

                            # Compute accuracy for this batch
                            preds = out[:batch_size].argmax(dim=1)
                            val_total_correct += (preds == y_batch[:batch_size]).sum().item()

                            # concatenate predictions and labels
                            all_preds.append(preds.cpu().detach().numpy())
                            all_labels.append(y_batch[:batch_size].cpu().detach().numpy())

                            # At the end of the loop:
                            del x_batch, y_batch, batch_edge_index, batch_edge_weight
                            if torch.cuda.is_available():
                                 torch.cuda.empty_cache()

                        val_acc = val_total_correct / val_total_samples
                        val_loss = val_total_loss / val_total_samples
                        pred_test = np.concatenate(all_preds, axis=0)
                        true_test = np.concatenate(all_labels, axis=0)

                    # calculate score
                    f1 = f1_score(true_test, pred_test, average='macro')
                    precision = precision_score(true_test,pred_test, average='macro', zero_division=1)
                    recall = recall_score(true_test,pred_test, average='macro',zero_division=1)
                    cm = confusion_matrix(true_test,pred_test)
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_val_loss = self.state_dict().copy()
                    if f1 > best_f1_score:
                        best_f1_score = f1
                        best_model_f1 = self.state_dict().copy()

                    # AUROC
                    num_classes = len(np.unique(true_test))
                    auroc_scores = []
                    for i in range(num_classes):
                        true_binary = (true_test == i)
                        pred_score = (pred_test == i)
                        try:
                            auroc = roc_auc_score(true_binary, pred_score)
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
                    print(f'Batch Epoch {index:>3} | Train Batch Loss: {train_loss:.3f} | Train Batch Acc: {acc * 100:>6.2f}% '
                          f'| Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}% '
                          f'| F1 score: {f1:.3f} | Precision: {precision:.3f} '
                          f'| Recall: {recall:.3f} | AUROC: {auroc:.3f} '
                          f'| Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f} | Confusion matrix: {cm}')

                    acc_list.append(acc)
                    loss_list.append(train_loss.cpu().item())
                    val_acc_list.append(val_acc)
                    val_loss_list.append(val_loss)
                    f1_list.append(f1)
                    auroc_list.append(auroc)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    sensitivity_list.append(sensitivity)
                    specificity_list.append(specificity)
                    cm_list.append(cm)
                    index_list.append(index)
                
                    
            
        # create a dictionary with column names as keys and lists as values
        trainning_data = {'Batch Epoch': index_list, 'Train_batch_acc': acc_list, 'Train_batch_loss': loss_list,
                          'Val_acc': val_acc_list, 'Val_loss': val_loss_list, 'F1_val': f1_list, 
                          'Precision_val': precision_list, 'Recall_val': recall_list, 
                          'Auroc':auroc_list, 'Sensitivity':sensitivity_list, 'Specificity':specificity_list, 'Confusion matrix': cm_list}

        result_df = pd.DataFrame(trainning_data)

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


        return result_df, best_model_val_loss, best_model_f1
