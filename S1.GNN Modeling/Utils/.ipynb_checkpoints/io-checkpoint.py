# We assume that PyTorch is already installed
import torch
torchversion = torch.__version__
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os
from Models import GAT_batch, GraphSAGE_batch, Magnet_batch, GCN_batch
from torch_geometric.loader import NeighborLoader
#from torch_geometric.loader import NeighborSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import umap
from sklearn.manifold import TSNE

from typing import Optional, Tuple
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_undirected

from typing import Union, Optional, Tuple
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add
import scipy
import math

from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes

from scipy.sparse.linalg import eigsh

#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def assign_race(df):
    # NHB:0  NHW:1 OT:2 UN:3
    conditions = []
    choices = []

    if '03' in df.columns:
        conditions.append(df['03'] == 1)
        choices.append(0)

    if '05' in df.columns:
        conditions.append(df['05'] == 1)
        choices.append(1)

    other_columns = ['OT', '01', '02', '04', '06', '07', 'NI']
    available_other_columns = [col for col in other_columns if col in df.columns]
    if available_other_columns:
        conditions.append(df[available_other_columns].eq(1).any(axis=1))
        choices.append(2)
        
    if 'UN' in df.columns:
        available_other_columns.append('UN')

    if conditions:
        df['Race'] = np.select(conditions, choices, default=3)
    else:
        df['Race'] = 3
        
    df.drop(available_other_columns + ['03','05'], axis=1, inplace=True)
    
    # Move 'CASE_RESULT' column next to 'M' column
    if 'Race' in df.columns:
        cols = list(df.columns)
        m_index = cols.index('M')
        cols.remove('Race')
        cols.insert(m_index + 1, 'Race')
        df = df.reindex(columns=cols)

    return df


def rev_sigmoid(x, k = 1):
    return 2*(1 / (1 + math.exp(k*x)))

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

def get_decomposition_method(method_name, n_components, random_state):
    if method_name == "PCA":
        return PCA(n_components=n_components, random_state=random_state)
    elif method_name == "UMAP":
        return umap.UMAP(n_components=n_components, random_state=random_state)
    elif method_name == "TSNE":
        return TSNE(n_components=n_components, random_state=random_state)
    else:
        print("Method not defined!!!")
        return None

def check_saving_path(save_dir, model_id):
    isExist = os.path.exists(save_dir)
    if isExist is False: os.makedirs(save_dir)
    save_path = os.path.join(save_dir, model_id)
    return save_path

def save_model(model, root_dir, models_dir, model_name, model_id):
    save_dir = os.path.join(root_dir, models_dir, model_name)
    save_path = check_saving_path(save_dir,model_id)
    torch.save(model, save_path)

def load_model(root_dir, models_dir, model_name, model_id):
    save_dir = os.path.join(root_dir, models_dir, model_name)
    isExist = os.path.exists(save_dir)
    if isExist is False:
        print("No folder exist!!!")
        return None
    save_path = os.path.join(save_dir, model_id)
    loaded_model = torch.load(save_path)
    return loaded_model

def save_numpy(numpy, root_dir, output_dir, model_name, model_id, filename):
    save_dir = os.path.join(root_dir, output_dir, model_name, model_id)
    save_path = check_saving_path(save_dir, filename)
    np.save(save_path, numpy)

def load_numpy(root_dir, output_dir, model_name, model_id, filename):
    save_dir = os.path.join(root_dir, output_dir, model_name, model_id)
    isExist = os.path.exists(save_dir)
    if isExist is False:
        print("No folder exist!!!")
        return None
    save_path = os.path.join(save_dir, filename)
    numpy_array = np.load(save_path)
    return numpy_array

def save_dataframe(dataframe, root_dir, output_log_dir, model_name, model_id, output_log_filename):
    save_dir = os.path.join(root_dir, output_log_dir,model_name, model_id)
    save_path = check_saving_path(save_dir, output_log_filename)
    dataframe.to_csv(save_path)

def save_fig(plt, root_dir, output_fig_dir, model_name, model_id, output_fig_filename):
    save_dir = os.path.join(root_dir, output_fig_dir,model_name, model_id)
    save_path = check_saving_path(save_dir, output_fig_filename)
    plt.savefig(save_path, dpi = 1200)

def load_dataframe(root_dir, output_dir, model_name, model_id, filename):
    save_dir = os.path.join(root_dir, output_dir, model_name, model_id)
    isExist = os.path.exists(save_dir)
    if isExist is False:
        print("No folder exist!!!")
        return None
    save_path = os.path.join(save_dir, filename)
    dataframe = pd.read_csv(save_path)
    return dataframe

def get_optimier(optimizer_type, model_parameters, lr, weight_decay):
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'Sgd':
        optimizer = torch.optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)

    return optimizer


def build_model(model_type, input_dim, hidden_dim, output_embedding_dim, output_dim, num_layers, dropout_prob, heads=1, batch_training = False
                , q = 0.25, K = 2, Magnet_activation = True):
    if batch_training:
        if model_type == 'GCN':
            model = GCN_batch.GCN_batch(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim = output_embedding_dim,
                                 output_dim=output_dim,num_layers=num_layers, dropout_prob=dropout_prob)
        elif model_type == 'GAT':
            model = GAT_batch.GAT_batch(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim=output_embedding_dim,
                      output_dim=output_dim, heads=heads, num_layers=num_layers, dropout_prob=dropout_prob)
        elif model_type == 'GraphSAGE':
            model = GraphSAGE_batch.GraphSAGE_batch(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim = output_embedding_dim,
                                 output_dim=output_dim, num_layers=num_layers, dropout_prob=dropout_prob)
        elif model_type == 'DIGCN':
            model = DIGCN_batch.DIGCN_batch(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim = output_embedding_dim,
                                 output_dim=output_dim, num_layers=num_layers, dropout_prob=dropout_prob)
        elif model_type == 'DGCN':
            model = DGCN_batch.DGCN_batch(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim = output_embedding_dim,
                                 output_dim=output_dim, num_layers=num_layers, dropout_prob=dropout_prob)
        elif model_type == 'Magnet':
            model = Magnet_batch.Magnet_batch(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim = output_embedding_dim,
                        output_dim=output_dim, num_layers=num_layers, dropout_prob=dropout_prob, q = q, K = K, activation = Magnet_activation)
        else:
            model = None
            print("Don't support this type of model")
        
    else:
        if model_type == 'GCN':
            model = GCN.GCN_model(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim = output_embedding_dim,
                                 output_dim=output_dim,num_layers=num_layers, dropout_prob=dropout_prob)
        elif model_type == 'GAT':
            model = GAT.GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim=output_embedding_dim,
                          output_dim=output_dim, heads=heads, num_layers=num_layers, dropout_prob=dropout_prob)
        elif model_type == 'GraphSAGE':
            model = GraphSAGE.GraphSAGE(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim = output_embedding_dim,
                                 output_dim=output_dim, num_layers=num_layers, dropout_prob=dropout_prob)
        elif model_type == 'DGCN':
            model = DGCN.DGCN_model(input_dim = input_dim, hidden_dim = hidden_dim,output_embedding_dim = output_embedding_dim,  output_dim = output_dim,
                              num_layers = num_layers, dropout_prob = dropout_prob)

        elif model_type == 'DIGCN':
            model = DIGCN.DIGCN_model(input_dim=input_dim, hidden_dim=hidden_dim, output_embedding_dim=output_embedding_dim,
                                output_dim=output_dim, num_layers=num_layers, dropout_prob=dropout_prob)
        else:
            model = None
            print("Don't support this type of model")

    return model




def load_data(path_dataset, path_adj_matrix, k):
    print(path_dataset)
    # loading files
    file_dataset = open(path_dataset, 'rb')
    file_adj_matrix = open(path_adj_matrix, 'rb')
    df_features = pickle.load(file_dataset)
    adj_matrix = pickle.load(file_adj_matrix)

    # change names make it consist 
    #df_features = df_features.rename(columns={'START_DATE': 'ADMIT_DATE', 'deid_pat_ID': 'PATID'})
    # drop off column and reset index
    # List of columns to drop
    columns_to_drop = ['ENCID', 'outcome_final_AD', 'PATID', 'ADMIT_DATE', 'next_status','(0.0, 49.0]']#, 'current_status'
    # Drop the columns if they exist
    features = df_features.drop(columns=columns_to_drop, errors='ignore')
    
    # reset index
    features.index = range(len(features))
    features = features.astype(np.float16)

    # Perform standard normalization on numerical features
    # find numerical features
    
    numerical_features = []
    for c in features.columns:
        if len(features[c].unique()) > 12:
            numerical_features.append(c)
        else:
            continue
            
    #Perform standard normalization
    if len(numerical_features) !=0 :
        scaler = MinMaxScaler()
        features[numerical_features] = scaler.fit_transform(features[numerical_features])
    

    # Loading edge info convert to coo format
    coo = adj_matrix.tocoo()
    weighted_edge_data = pd.DataFrame({'source': coo.row, 'target': coo.col, 'weight': coo.data})
    
    # convert edge weight to 0-1
    for i in range(len(coo.data)) :
        coo.data[i] = rev_sigmoid(coo.data[i], k = k)

    
    # loading edge_weight and edge_index
    row = torch.from_numpy(coo.row.astype(np.int32)).type(torch.long)
    col = torch.from_numpy(coo.col.astype(np.int32)).type(torch.long)
    edge_weight = torch.from_numpy(coo.data.astype(np.float16)).type(torch.float)
    edge_index = torch.stack([row, col], dim=0)

    # loading the label from the dataset
    label = df_features['next_status'].astype(int)
    label.index = range(len(label))
    
    df_features.reset_index(inplace=True)

    # get unique PATID list
    #PATID_list = df_features['PATID']  # dataframe

    return  features, label, edge_index, edge_weight, df_features


def split_data_by_PATID(PATID_list, val_ratio, features, label, edge_index, edge_weight):
    # get unique PATID and splite the based on PATID
    unique_PATID_list = PATID_list.unique()
    train_ratio = 0.7
    validation_ratio = 0.1
    test_ratio = 0.2
    
    # First split to separate out the training set
    train_pat, temp_set = train_test_split(unique_PATID_list, test_size=(1 - train_ratio), random_state=42)
    
    # Second split to divide the temp_set into validation and test sets
    val_test_ratio = test_ratio / (validation_ratio + test_ratio)
    val_pat, test_pat = train_test_split(temp_set, test_size=val_test_ratio, random_state=42)

    # filter patid_df get index based on splitted train PATID and test PATID
    patid_df = PATID_list.to_frame()
    patid_df.index = range(len(patid_df))
    train_df = patid_df[patid_df['PATID'].isin(train_pat)]
    val_df = patid_df[patid_df['PATID'].isin(val_pat)]
    test_df = patid_df[patid_df['PATID'].isin(test_pat)]

    # Get the train and test index for creat mask in the dataset
    X_train_index = torch.from_numpy(train_df.index.values.astype(np.int32)).type(torch.long)
    X_val_index = torch.from_numpy(val_df.index.values.astype(np.int32)).type(torch.long)
    X_test_index = torch.from_numpy(test_df.index.values.astype(np.int32)).type(torch.long)

    # create a dictionary to map old indices to new ones
    train_index_map = dict(zip(X_train_index.tolist(), range(len(X_train_index))))
    val_index_map = dict(zip(X_val_index.tolist(), range(len(X_val_index))))
    test_index_map = dict(zip(X_test_index.tolist(), range(len(X_test_index))))

    # For traning part
    # makesure all edge index pair in X_train_index
    mask_0 = torch.isin(edge_index[0], X_train_index)
    mask_1 = torch.isin(edge_index[1], X_train_index)
    mask = torch.logical_and(mask_0, mask_1)
    # filter train
    train_edge_index = edge_index[:, mask]
    train_edge_weight = edge_weight[mask]
    train_data_x = features.values[X_train_index]
    train_data_y = label.values[X_train_index]
    # map the old indices in filtered_edge_index to the new ones
    mapped_train_edge_index = torch.tensor(
        [[train_index_map[x.item()] for x in pair] for pair in train_edge_index])
    # save in data object
    train_data = Data(edge_index=mapped_train_edge_index, edge_weight=train_edge_weight)
    train_data.x = torch.tensor(train_data_x).type(torch.float)
    # loading labels
    y = torch.tensor(train_data_y)
    train_data.y = y.clone().detach()

    # For Val part
    # makesure all edge index pair in X_val_index
    mask_0 = torch.isin(edge_index[0], X_val_index)
    mask_1 = torch.isin(edge_index[1], X_val_index)
    mask = torch.logical_and(mask_0, mask_1)
    # filter val
    val_edge_index = edge_index[:, mask]
    val_edge_weight = edge_weight[mask]
    val_data_x = features.values[X_val_index]
    val_data_y = label.values[X_val_index]
    # map the old indices in filtered_edge_index to the new ones
    mapped_val_edge_index = torch.tensor([[val_index_map[x.item()] for x in pair] for pair in val_edge_index])
    # save in data object
    val_data = Data(edge_index=mapped_val_edge_index, edge_weight=val_edge_weight)
    val_data.x = torch.tensor(val_data_x).type(torch.float)
    # loading labels
    y = torch.tensor(val_data_y)
    val_data.y = y.clone().detach()
    
    # For Test part
    # makesure all edge index pair in X_val_index
    mask_0 = torch.isin(edge_index[0], X_test_index)
    mask_1 = torch.isin(edge_index[1], X_test_index)
    mask = torch.logical_and(mask_0, mask_1)
    # filter val
    test_edge_index = edge_index[:, mask]
    test_edge_weight = edge_weight[mask]
    test_data_x = features.values[X_test_index]
    test_data_y = label.values[X_test_index]
    # map the old indices in filtered_edge_index to the new ones
    mapped_test_edge_index = torch.tensor([[test_index_map[x.item()] for x in pair] for pair in test_edge_index])
    # save in data object
    test_data = Data(edge_index=mapped_test_edge_index, edge_weight=test_edge_weight)
    test_data.x = torch.tensor(test_data_x).type(torch.float)
    # loading labels
    y = torch.tensor(test_data_y)
    test_data.y = y.clone().detach()

    print('#Training samples :', len(train_df))
    print('Training percent :', len(train_df) / len(patid_df))
    print('#Val samples :', len(val_df))
    print('Val percent :', len(val_df) / len(patid_df))
    print('#Test samples :', len(test_df))
    print('Test percent :', len(test_df) / len(patid_df))

    return train_data, val_data, test_data


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


def directed_features_in_out(edge_index: torch.LongTensor, size: int,
                             edge_weight: Optional[torch.FloatTensor] = None, device: str = 'cuda:0') -> Tuple[torch.LongTensor, torch.LongTensor,
                                                                                                            torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
    r""" Computes directed in-degree and out-degree features.
    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **size** (int or None) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **device** (str, optional) - The device to store the returned values. (default: :str:`cpu`)
    Return types:
        * **index_undirected** (PyTorch LongTensor) - Undirected edge_index.
        * **edge_in** (PyTorch LongTensor) - Inwards edge indices.
        * **in_weight** (PyTorch Tensor) - Inwards edge weights.
        * **edge_out** (PyTorch LongTensor) - Outwards edge indices.
        * **out_weight** (PyTorch Tensor) - Outwards edge weights.
    """
    if edge_weight is not None:
        a = sp.coo_matrix((edge_weight.cpu(), edge_index.cpu()),
                          shape=(size, size)).tocsc()
    else:
        a = sp.coo_matrix(
            (np.ones(len(edge_index[0])), edge_index.cpu()), shape=(size, size)).tocsc()

    out_degree = np.array(a.sum(axis=0))[0]
    out_degree[out_degree == 0] = 1

    in_degree = np.array(a.sum(axis=1))[:, 0]
    in_degree[in_degree == 0] = 1

    # sparse implementation
    a = sp.csr_matrix(a)
    A_in = sp.csr_matrix(np.zeros((size, size)))
    A_out = sp.csr_matrix(np.zeros((size, size)))
    for k in range(size):
        A_in += np.dot(a[k, :].T, a[k, :])/out_degree[k]
        A_out += np.dot(a[:, k], a[:, k].T)/in_degree[k]

    A_in = A_in.tocoo()
    A_out = A_out.tocoo()

    edge_in = torch.from_numpy(np.vstack((A_in.row,  A_in.col))).long()
    edge_out = torch.from_numpy(np.vstack((A_out.row, A_out.col))).long()

    in_weight = torch.from_numpy(A_in.data).float()
    out_weight = torch.from_numpy(A_out.data).float()
    index_undirected = to_undirected(edge_index)

    device = edge_index.device
    return index_undirected.to(device), edge_in.to(device), in_weight.to(device), edge_out.to(device), out_weight.to(device)



def get_appr_directed_adj(alpha: float, edge_index: torch.LongTensor,
                          num_nodes: Union[int, None], dtype: torch.dtype,
                          edge_weight: Optional[torch.FloatTensor] = None) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    r""" Computes the approximate pagerank adjacency matrix of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the
    `Digraph Inception Convolutional Networks
    <https://papers.nips.cc/paper/2020/file/cffb6e2288a630c2a787a64ccc67097c-Paper.pdf>`_ paper.
    Arg types:
        * **alpha** (float) -alpha used in approximate personalized page rank.
        * **edge_index** (PyTorch LongTensor) -The edge indices.
        * **num_nodes** (int or None) -The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        * **dtype** (torch.dtype) -The desired data type of returned tensor in case :obj:`edge_weight=None`.
        * **edge_weight** (PyTorch Tensor, optional) -One-dimensional edge weights. (default: :obj:`None`)
    Return types:
        * **edge_index** (PyTorch LongTensor) -The edge indices of the approximate page-rank matrix.
        * **edge_weight** (PyTorch Tensor) -One-dimensional edge weights of the approximate page-rank matrix.
    """
    if edge_weight == None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(
        edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1, num_nodes+1]))
    p_v[0:num_nodes, 0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes, 0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes, num_nodes] = alpha
    p_v[num_nodes, num_nodes] = 0.0
    p_ppr = p_v

    eig_value, left_vector = scipy.linalg.eig(
        p_ppr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    _, ind = eig_value.sort(descending=True)

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi
    # print(pi)
    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    pi_sqrt = pi_sqrt.to(p_ppr.device)
    pi_inv_sqrt = pi_inv_sqrt.to(p_ppr.device)
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) +
         torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0
    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()

    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def proprocess_directed_adj(train_data, val_data, alpha):
    print("Preprocessing Training garph")
    train_data.edge_index, train_data.edge_weight = get_appr_directed_adj(alpha,
                                                                                train_data.edge_index,
                                                                                train_data.x.shape[0],
                                                                                train_data.x.dtype,
                                                                                train_data.edge_weight)
    print("Preprocessing Val garph")
    val_data.edge_index, val_data.edge_weight = get_appr_directed_adj(alpha,
                                                                            val_data.edge_index,
                                                                            val_data.x.shape[0],
                                                                            val_data.x.dtype,
                                                                            val_data.edge_weight)

    return train_data, val_data


def get_magnetic_Laplacian(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                           normalization: Optional[str] = 'sym',
                           dtype: Optional[int] = None,
                           num_nodes: Optional[int] = None,
                           q: Optional[float] = 0.25,
                           return_lambda_max: bool = False):
    r""" Computes the magnetic Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight` from the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.

    Arg types:
        * **edge_index** (PyTorch LongTensor) - The edge indices.
        * **edge_weight** (PyTorch Tensor, optional) - One-dimensional edge weights. (default: :obj:`None`)
        * **normalization** (str, optional) - The normalization scheme for the magnetic Laplacian (default: :obj:`sym`) -

            1. :obj:`None`: No normalization :math:`\mathbf{L} = \mathbf{D} - \mathbf{A} Hadamard \exp(i \Theta^{(q)})`

            2. :obj:`"sym"`: Symmetric normalization :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2} Hadamard \exp(i \Theta^{(q)})`

        * **dtype** (torch.dtype, optional) - The desired data type of returned tensor in case :obj:`edge_weight=None`. (default: :obj:`None`)
        * **num_nodes** (int, optional) - The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        * **q** (float, optional) - The value q in the paper for phase.
        * **return_lambda_max** (bool, optional) - Whether to return the maximum eigenvalue. (default: :obj:`False`)

    Return types:
        * **edge_index** (PyTorch LongTensor) - The edge indices of the magnetic Laplacian.
        * **edge_weight.real, edge_weight.imag** (PyTorch Tensor) - Real and imaginary parts of the one-dimensional edge weights for the magnetic Laplacian.
        * **lambda_max** (float, optional) - The maximum eigenvalue of the magnetic Laplacian, only returns this when required by setting return_lambda_max as True.
    """

    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    edge_attr = torch.stack([sym_attr, theta_attr], dim=1)

    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                         num_nodes, "add")

    edge_weight_sym = edge_attr[:, 0]
    edge_weight_sym = edge_weight_sym/2

    row, col = edge_index_sym[0], edge_index_sym[1]
    deg = scatter_add(edge_weight_sym, row, dim=0, dim_size=num_nodes)

    edge_weight_q = torch.exp(1j * 2 * np.pi * q * edge_attr[:, 1])

    if normalization is None:
        # L = D_sym - A_sym Hadamard \exp(i \Theta^{(q)}).
        edge_index, _ = add_self_loops(edge_index_sym, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight_sym * edge_weight_q, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = D_sym^{-1/2} A_sym D_sym^{-1/2} Hadamard \exp(i \Theta^{(q)}).
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * \
            edge_weight_sym * deg_inv_sqrt[col] * edge_weight_q

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index_sym, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp
    if not return_lambda_max:
        return edge_index, edge_weight.real, edge_weight.imag
    else:
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)
        lambda_max = float(lambda_max.real)
        return edge_index, edge_weight.real, edge_weight.imag, lambda_max
    

def embedding_batch(model, data, batch_size, device):
    # Set the model to evaluation mode
    model.eval()

    # Create a NeighborSampler for evaluation
    loader = NeighborLoader(data, num_neighbors=[-1]*model.num_layers, batch_size=batch_size, shuffle=False)
    
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            # push every thing to device
            batch_size = batch.batch_size
            x_batch = batch.x.to(device)
            y_batch = batch.y.to(device)
            batch_edge_index = batch.edge_index.to(device)



            # Get embeddings and logits for this batch
            embeddings_batch, logits_batch = model(x_batch, batch_edge_index)

            # Only keep the embeddings for the nodes in the batch (without neighbors)
            all_embeddings.append(embeddings_batch[:batch_size].cpu())

            # Compute loss for this batch
            loss = criterion(logits_batch[:batch_size], y_batch[:batch_size])
            total_loss += loss.item() * batch_size

            # Compute accuracy for this batch
            preds = logits_batch[:batch_size].argmax(dim=1)
            total_correct += (preds == y_batch[:batch_size]).sum().item()
            # which should be all node in batch
            total_samples += batch_size

            # At the end of the loop:
            del x_batch, y_batch, batch_edge_index
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Compute overall accuracy and loss
    overall_acc = total_correct / total_samples
    overall_loss = total_loss / total_samples

    print(f'On the whole dataset: Accuracy: {overall_acc:.4f}, Loss: {overall_loss:.4f}')
    
    return all_embeddings.numpy()


def embedding_batch_DIGCN(model, data, batch_size, device, alpha):
    # Set the model to evaluation mode
    model.eval()

    # Create a NeighborSampler for evaluation
    loader = NeighborLoader(data, num_neighbors=[-1]*model.num_layers, batch_size=batch_size, shuffle=False)
    
    #preprocess
    print('start preprocess data')
    for batch in loader:
        batch.edge_index, batch.edge_weight = get_appr_directed_adj(alpha, batch.edge_index, batch.x.shape[0], batch.x.dtype, batch.edge_weight)
    
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_embeddings = []

    with torch.no_grad():
        for batch in loader:

            # push every thing to device
            batch_size = batch.batch_size
            x_batch = batch.x.to(device)
            y_batch = batch.y.to(device)
            batch_edge_index = batch.edge_index.to(device)
            batch_edge_weight = batch.edge_weight.to(device)

            # Get embeddings and logits for this batch
            embeddings_batch, logits_batch = model(x_batch, batch_edge_index, batch_edge_weight)

            # Only keep the embeddings for the nodes in the batch (without neighbors)
            all_embeddings.append(embeddings_batch[:batch_size].cpu())

            # Compute loss for this batch
            loss = criterion(logits_batch[:batch_size], y_batch[:batch_size])
            total_loss += loss.item() * batch_size

            # Compute accuracy for this batch
            preds = logits_batch[:batch_size].argmax(dim=1)
            total_correct += (preds == y_batch[:batch_size]).sum().item()
            # which should be all node in batch
            total_samples += batch_size

            # At the end of the loop:
            del x_batch, y_batch, batch_edge_index, batch_edge_weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    # Compute overall accuracy and loss
    overall_acc = total_correct / total_samples
    overall_loss = total_loss / total_samples

    print(f'On the whole dataset: Accuracy: {overall_acc:.4f}, Loss: {overall_loss:.4f}')
    
    return all_embeddings.numpy()



def embedding_batch_Magnet(model, data, batch_size, device):
    # Set the model to evaluation mode
    model.eval()

    # Create a NeighborSampler for evaluation
    loader = NeighborLoader(data, num_neighbors=[-1]*model.num_layers, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_embeddings = []

    with torch.no_grad():
        for batch in loader:
            # push every thing to device
            batch_size = batch.batch_size
            x_batch = batch.x.to(device)
            y_batch = batch.y.to(device)
            batch_edge_index = batch.edge_index.to(device)
            batch_edge_weight = batch.edge_weight.to(device)

            # Get embeddings and logits for this batch
            embeddings_batch, logits_batch = model(x_batch, x_batch, batch_edge_index, batch_edge_weight)

            # Only keep the embeddings for the nodes in the batch (without neighbors)
            all_embeddings.append(embeddings_batch[:batch_size].cpu().detach())

            # Compute loss for this batch
            loss = criterion(logits_batch[:batch_size], y_batch[:batch_size])
            total_loss += loss.item() * batch_size

            # Compute accuracy for this batch
            preds = logits_batch[:batch_size].argmax(dim=1)
            total_correct += (preds == y_batch[:batch_size]).sum().item()
            # which should be all node in batch
            total_samples += batch_size

            # At the end of the loop:
            del x_batch, y_batch, batch_edge_index, batch_edge_weight, embeddings_batch, logits_batch 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    # Compute overall accuracy and loss
    overall_acc = total_correct / total_samples
    overall_loss = total_loss / total_samples

    print(f'On the whole dataset: Accuracy: {overall_acc:.4f}, Loss: {overall_loss:.4f}')
    
    return all_embeddings.numpy()


def embedding_batch_DGCN(model, data, batch_size, device):
    # Set the model to evaluation mode
    model.eval()

    # Create a NeighborSampler for evaluation
    loader =  NeighborLoader(data, num_neighbors=[-1]*model.num_layers, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_embeddings = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # push every thing to device
            batch_size = batch.batch_size
            x_batch = batch.x.to(device)
            y_batch = batch.y.to(device)
            batch_edge_index = batch.edge_index.to(device)
            batch_edge_weight = batch.edge_weight.to(device)
            
            #preprocess
            edge_index, edge_in, in_weight, edge_out, out_weight = directed_features_in_out(batch_edge_index, len(batch.x), batch_edge_weight)

            # Get embeddings and logits for this batch
            embeddings_batch, out = self(x_batch, edge_index, edge_in, in_weight, edge_out, out_weight)

            # Only keep the embeddings for the nodes in the batch (without neighbors)
            all_embeddings.append(embeddings_batch[:batch_size].cpu())
            
            # only using batch node for calculate loss
            loss = criterion(out[:batch_size], y_batch[:batch_size])

            total_samples += batch_size
            total_loss += float(loss) * batch_size

            # Compute accuracy for this batch
            preds = out[:batch_size].argmax(dim=1)
            total_correct += (preds == y_batch[:batch_size]).sum().item()

            # At the end of the loop:
            del x_batch, y_batch, batch_edge_index, batch_edge_weight, edge_index, edge_in, in_weight, edge_out, out_weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Compute overall accuracy and loss
    overall_acc = total_correct / total_samples
    overall_loss = total_loss / total_samples

    print(f'On the whole dataset: Accuracy: {overall_acc:.4f}, Loss: {overall_loss:.4f}')
    
    return all_embeddings.numpy()

def embedding_batch_GCN(model, data, batch_size, device):
    # Set the model to evaluation mode
    model.eval()

    # Create a NeighborSampler for evaluation
    loader = NeighborLoader(data, num_neighbors=[-1]*model.num_layers, batch_size=batch_size, shuffle=False)
    
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_embeddings = []

    with torch.no_grad():
        for batch in loader:

            # push every thing to device
            batch_size = batch.batch_size
            x_batch = batch.x.to(device)
            y_batch = batch.y.to(device)
            batch_edge_index = batch.edge_index.to(device)
            batch_edge_weight = batch.edge_weight.to(device)

            # Get embeddings and logits for this batch
            embeddings_batch, logits_batch = model(x_batch, batch_edge_index, batch_edge_weight)

            # Only keep the embeddings for the nodes in the batch (without neighbors)
            all_embeddings.append(embeddings_batch[:batch_size].cpu())

            # Compute loss for this batch
            loss = criterion(logits_batch[:batch_size], y_batch[:batch_size])
            total_loss += loss.item() * batch_size

            # Compute accuracy for this batch
            preds = logits_batch[:batch_size].argmax(dim=1)
            total_correct += (preds == y_batch[:batch_size]).sum().item()
            # which should be all node in batch
            total_samples += batch_size

            # At the end of the loop:
            del x_batch, y_batch, batch_edge_index, batch_edge_weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu()

    # Compute overall accuracy and loss
    overall_acc = total_correct / total_samples
    overall_loss = total_loss / total_samples

    print(f'On the whole dataset: Accuracy: {overall_acc:.4f}, Loss: {overall_loss:.4f}')
    
    return all_embeddings.numpy()


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
    data = Data(x = torch.tensor(features.values).type(torch.float), y=torch.tensor(label.values), edge_index=edge_index, edge_weight=edge_weight)
    # loading data to GPU or CPU
    data = data.to(device)
    data.is_cuda
    print(data.num_features)
    data.num_nodes = len(features)
    data.num_classes = len(label.unique())

    # get unique PATID list
    PATID_list = df_features['PATID']
    #split dataset to subgraph
    train_data, val_data = split_data_by_PATID(PATID_list, 0.3, features, label, edge_index, edge_weight)

    import time
    print("start!!!")
    start_time = time.time()
    data.edge_index, data.edge_weight = get_appr_directed_adj(0.1, data.edge_index, data.x.shape[0], data.x.dtype,
                                                                    data.edge_weight)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f'The function took {elapsed_time:.2f} seconds to execute.')
    """