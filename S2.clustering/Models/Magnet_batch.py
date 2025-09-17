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
import random
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from Utils import io as Utils
#from torch_geometric.loader import NeighborSampler (old verison)
from torch_geometric.loader import NeighborLoader
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.utils import remove_self_loops, add_self_loops

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

import torch
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

class complex_relu_layer(nn.Module):
    """The complex ReLU layer from the `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.
    """

    def __init__(self, ):
        super(complex_relu_layer, self).__init__()

    def complex_relu(self, real: torch.FloatTensor, img: torch.FloatTensor):
        """
        Complex ReLU function.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real: torch.FloatTensor, img: torch.FloatTensor):
        """
        Making a forward pass of the complex ReLU layer.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        real, img = self.complex_relu(real, img)
        return real, img

class MagNetConv(MessagePassing):
    r"""The magnetic graph convolutional operator from the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper
    :math:`\mathbf{\hat{L}}` denotes the scaled and normalized magnetic Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        K (int): Chebyshev filter size :math:`K`.
        q (float, optional): Initial value of the phase parameter, 0 <= q <= 0.25. Default: 0.25.
        trainable_q (bool, optional): whether to set q to be trainable or not. (default: :obj:`False`)
        normalization (str, optional): The normalization scheme for the magnetic
            Laplacian (default: :obj:`sym`):
            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} \odot \exp(i \Theta^{(q)})`
            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} \odot \exp(i \Theta^{(q)})`
            `\odot` denotes the element-wise multiplication.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the __norm__ matrix on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, q: float, trainable_q: bool = False,
                 normalization: str = 'sym', cached: bool = False, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(MagNetConv, self).__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym'], 'Invalid normalization'
        kwargs.setdefault('flow', 'target_to_source')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.cached = cached
        self.trainable_q = trainable_q
        if trainable_q:
            self.q = Parameter(torch.Tensor(1).fill_(q))
        else:
            self.q = q
        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))

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
        self.cached_q = None

    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        q: float,
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None
    ):
        """
        Get magnetic laplacian.

        Arg types:
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * num_nodes (int, Optional) - Node features.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.

        Return types:
            * edge_index_real, edge_index_imag, edge_weight_real, edge_weight_imag (PyTorch Float Tensor) - Magnetic laplacian tensor: real and imaginary edge indices and weights.
        """
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight_real, edge_weight_imag = Utils.get_magnetic_Laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes, q
        )

        edge_weight_real = (2.0 * edge_weight_real) / lambda_max
        edge_weight_real.masked_fill_(edge_weight_real == float("inf"), 0)
        edge_index_imag = edge_index.clone()

        edge_index_real, edge_weight_real = add_self_loops(
            edge_index, edge_weight_real, fill_value=-1.0, num_nodes=num_nodes
        )
        assert edge_weight_real is not None

        edge_weight_imag = (2.0 * edge_weight_imag) / lambda_max
        edge_weight_imag.masked_fill_(edge_weight_imag == float("inf"), 0)

        assert edge_weight_imag is not None

        return edge_index_real, edge_index_imag, edge_weight_real, edge_weight_imag

    def forward(
        self,
        x_real: torch.FloatTensor,
        x_imag: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the MagNet Convolution layer.

        Arg types:
            * x_real, x_imag (PyTorch Float Tensor) - Node features.
            * edge_index (PyTorch Long Tensor) - Edge indices.
            * edge_weight (PyTorch Float Tensor, optional) - Edge weights corresponding to edge indices.
            * lambda_max (optional, but mandatory if normalization is None) - Largest eigenvalue of Laplacian.
        Return types:
            * out_real, out_imag (PyTorch Float Tensor) - Hidden state tensor for all nodes, with shape (N_nodes, F_out).
        """
        if self.trainable_q:
            self.q = Parameter(torch.clamp(self.q, 0, 0.25))

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
            if self.q != self.cached_q:
                raise RuntimeError(
                    'Cached q is {}, but found {} in input. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_q, self.q))
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.trainable_q:
                self.cached_q = self.q.detach().item()
            else:
                self.cached_q = self.q
            if self.normalization != 'sym' and lambda_max is None:
                if self.trainable_q:
                    raise RuntimeError(
                        'Cannot train q while not calculating maximum eigenvalue of Laplacian!')
                _, _, _, lambda_max = Utils.get_magnetic_Laplacian(
                    edge_index, edge_weight, None, q=self.q, return_lambda_max=True
                )

            if lambda_max is None:
                lambda_max = torch.tensor(
                    2.0, dtype=x_real.dtype, device=x_real.device)
            if not isinstance(lambda_max, torch.Tensor):
                lambda_max = torch.tensor(lambda_max, dtype=x_real.dtype,
                                          device=x_real.device)
            assert lambda_max is not None
            edge_index_real, edge_index_imag, norm_real, norm_imag = self.__norm__(edge_index, x_real.size(self.node_dim),
                                                             edge_weight, self.q, self.normalization,
                                                             lambda_max, dtype=x_real.dtype)
            self.cached_result = edge_index_real, edge_index_imag, norm_real, norm_imag

        edge_index_real, edge_index_imag, norm_real, norm_imag = self.cached_result

        Tx_0_real_real = x_real
        Tx_0_imag_imag = x_imag
        Tx_0_imag_real = x_real
        Tx_0_real_imag = x_imag
        out_real_real = torch.matmul(Tx_0_real_real, self.weight[0])
        out_imag_imag = torch.matmul(Tx_0_imag_imag, self.weight[0])
        out_imag_real = torch.matmul(Tx_0_imag_real, self.weight[0])
        out_real_imag = torch.matmul(Tx_0_real_imag, self.weight[0])

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.size(0) > 1:
            Tx_1_real_real = self.propagate(
                edge_index_real, x=x_real, norm=norm_real, size=None)
            out_real_real = out_real_real + \
                torch.matmul(Tx_1_real_real, self.weight[1])
            Tx_1_imag_imag = self.propagate(
                edge_index_imag, x=x_imag, norm=norm_imag, size=None)
            out_imag_imag = out_imag_imag + \
                torch.matmul(Tx_1_imag_imag, self.weight[1])
            Tx_1_imag_real = self.propagate(
                edge_index_real, x=x_real, norm=norm_real, size=None)
            out_imag_real = out_imag_real + \
                torch.matmul(Tx_1_imag_real, self.weight[1])
            Tx_1_real_imag = self.propagate(
                edge_index_imag, x=x_imag, norm=norm_imag, size=None)
            out_real_imag = out_real_imag + \
                torch.matmul(Tx_1_real_imag, self.weight[1])

        for k in range(2, self.weight.size(0)):
            Tx_2_real_real = self.propagate(
                edge_index_real, x=Tx_1_real_real, norm=norm_real, size=None)
            Tx_2_real_real = 2. * Tx_2_real_real - Tx_0_real_real
            out_real_real = out_real_real + \
                torch.matmul(Tx_2_real_real, self.weight[k])
            Tx_0_real_real, Tx_1_real_real = Tx_1_real_real, Tx_2_real_real

            Tx_2_imag_imag = self.propagate(
                edge_index_imag, x=Tx_1_imag_imag, norm=norm_imag, size=None)
            Tx_2_imag_imag = 2. * Tx_2_imag_imag - Tx_0_imag_imag
            out_imag_imag = out_imag_imag + \
                torch.matmul(Tx_2_imag_imag, self.weight[k])
            Tx_0_imag_imag, Tx_1_imag_imag = Tx_1_imag_imag, Tx_2_imag_imag

            Tx_2_imag_real = self.propagate(
                edge_index_real, x=Tx_1_imag_real, norm=norm_real, size=None)
            Tx_2_imag_real = 2. * Tx_2_imag_real - Tx_0_imag_real
            out_imag_real = out_imag_real + \
                torch.matmul(Tx_2_imag_real, self.weight[k])
            Tx_0_imag_real, Tx_1_imag_real = Tx_1_imag_real, Tx_2_imag_real

            Tx_2_real_imag = self.propagate(
                edge_index_imag, x=Tx_1_real_imag, norm=norm_imag, size=None)
            Tx_2_real_imag = 2. * Tx_2_real_imag - Tx_0_real_imag
            out_real_imag = out_real_imag + \
                torch.matmul(Tx_2_real_imag, self.weight[k])
            Tx_0_real_imag, Tx_1_real_imag = Tx_1_real_imag, Tx_2_real_imag

        out_real = out_real_real - out_imag_imag
        out_imag = out_imag_real + out_real_imag

        if self.bias is not None:
            out_real += self.bias
            out_imag += self.bias

        return out_real, out_imag

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)
    
#------------------------------------------Magnet model class-----------------------------------------------------

class Magnet_batch(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_embedding_dim, output_dim, num_layers, dropout_prob, q, K, activation):
        
        super(Magnet_batch, self).__init__()

        self.num_layers = num_layers
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu_layer()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(MagNetConv(in_channels=input_dim, out_channels= hidden_dim, K=K,
                                    q=q))
        for i in range(num_layers - 2):
            self.convs.append(MagNetConv(in_channels=hidden_dim, out_channels= hidden_dim, K=K,
                                    q=q))
        self.convs.append(MagNetConv(in_channels=hidden_dim, out_channels= hidden_dim, K=K,
                                    q=q))
                          
        self.Conv = nn.Conv1d(2*hidden_dim, output_embedding_dim, kernel_size=1)
        self.fc = nn.Linear(output_embedding_dim, output_dim)
        self.dropout_prob = dropout_prob

    def forward(self, real, imag, edge_index, edge_weight):
                          
        for i in range(self.num_layers):
            real, imag = self.convs[i](real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)
                          
        x = torch.cat((real, imag), dim=-1)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
                          
        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        
        x = torch.transpose(x[0], 0, 1)
        
        y = self.fc(x)
        return x, F.log_softmax(y, dim=1)
    
    def fit(self,  training_data, val_data, epochs, optimizer, batch_size, device, batch_log_size, batch_model_save_number, output_root_dir , models_dir, model_name, model_id, gamma):
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = FocalLoss(gamma=gamma)
        print('Focal loss gamma: ',gamma)
        
        train_loader = NeighborLoader(training_data, num_neighbors=[-1]*self.num_layers, batch_size=batch_size, shuffle =True)
        val_loader = NeighborLoader(val_data, num_neighbors=[-1]*self.num_layers, batch_size=batch_size)
        print('number of batch in trainning data: ', len(train_loader))
        

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
            
                embedding, out = self(x_batch, x_batch, batch_edge_index, batch_edge_weight)
                
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
                #save the model in every batch_model_save_number batches
                if index % batch_model_save_number == 1:
                    current_model = self.state_dict().copy()
                    save_dir = os.path.join(output_root_dir, models_dir, model_name)
                    isExist = os.path.exists(save_dir)
                    if isExist is False: os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, model_id)
                    torch.save(current_model, save_path+'batch_index_'+str(index))
                
                if index%batch_log_size == 1:
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

                            embedding, out = self(x_batch, x_batch, batch_edge_index, batch_edge_weight)

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
                    
        # save the last model
        current_model = self.state_dict().copy()
        save_dir = os.path.join(output_root_dir, models_dir, model_name)
        isExist = os.path.exists(save_dir)
        if isExist is False: os.makedirs(save_dir)
        save_path = os.path.join(save_dir, model_id)
        torch.save(current_model, save_path+'batch_last')
            
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

if __name__ == "__main__":
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

    DIGCNmodel = DIGCN_model(input_dim = input_dim, hidden_dim = hidden_dim,output_embedding_dim = output_embedding_dim,  output_dim = output_dim,
                          num_layers = num_GCN_layers, dropout_prob = dropout_prob)
    DIGCNmodel.to(device)

    print(DIGCNmodel)

    # Define the optimizers to use
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(DIGCNmodel.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'Sgd':
        optimizer = torch.optim.SGD(DIGCNmodel.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(DIGCNmodel.parameters(), lr=learning_rate, weight_decay=weight_decay)

    import time
    print("start!!!")
    start_time = time.time()

    train_data, val_data = Utils.proprocess_directed_adj(train_data = train_data, val_data = val_data, alpha=0.1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'The function took {elapsed_time:.2f} seconds to execute.')

    DIGCNmodel.fit(train_data, val_data, epochs=epochs, optimizer=optimizer)

    data.edge_index, data.edge_weight = Utils.get_appr_directed_adj(0.1, data.edge_index, data.x.shape[0],data.x.dtype,data.edge_weight)


    output_from_model = DIGCNmodel(data.x,data.edge_index, data.edge_weight)
    embedding_feature = output_from_model[0].cpu().detach().numpy()
    print(embedding_feature.shape)
    print(Utils.accuracy(output_from_model[1].argmax(dim=1),data.y))
    """