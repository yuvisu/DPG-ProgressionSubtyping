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
#from torch_geometric.loader import NeighborSampler (old verison)
from torch_geometric.loader import NeighborLoader
import random
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

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
#device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

class GraphSAGE_batch(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_embedding_dim, output_dim, num_layers, dropout_prob):
        super(GraphSAGE_batch, self).__init__()

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
    """
    
    def forward(self, x, adjs):
        for i in range(self.num_layers - 1):
            x = F.relu(self.sages[i](x=x, edge_index=adjs[i].edge_index))
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
        x = self.sages[self.num_layers - 1](x=x, edge_index=adjs[-1].edge_index)
        y = self.fc(x)
        return x, F.log_softmax(y, dim=1)
    """
    def fit(self,  training_data, val_data, epochs, optimizer, batch_size, device, batch_log_size, batch_model_save_number, output_root_dir , models_dir, model_name, model_id, gamma):
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = FocalLoss(gamma=gamma)
        print('Focal loss gamma: ',gamma)
        
        best_val_loss = float('inf')
        best_f1_score = 0
        best_model_val_loss = None
        best_model_f1 = None
        
        train_loader = NeighborLoader(training_data, num_neighbors=[-1]*self.num_layers, batch_size=batch_size, shuffle =True)
        val_loader = NeighborLoader(val_data, num_neighbors=[-1]*self.num_layers, batch_size=batch_size)
        print('number of batch in trainning data: ', len(train_loader))
        
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
                

                embedding, out = self(x_batch, batch_edge_index)
                
                train_loss = criterion(out, y_batch)
                #acc = accuracy(out.argmax(dim=1), y_batch)
                train_loss.backward()
                optimizer.step()
                
                train_total_samples += batch_size
                train_total_loss += float(train_loss) * batch_size
                
                # Compute accuracy for this batch
                preds = out[:batch_size].argmax(dim=1)
                #train_total_correct += (preds == y_batch[:batch_size]).sum().item()
                batch_total_correct = (preds == y_batch[:batch_size]).sum().item()

                # At the end of the loop:
                #del x_batch, y_batch, batch_edge_index
                #if torch.cuda.is_available():
                #    torch.cuda.empty_cache()
                    
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

                            embedding, out = self(x_batch, batch_edge_index)

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
                           # del x_batch, y_batch, batch_edge_index
                            #if torch.cuda.is_available():
                             #    torch.cuda.empty_cache()

                        val_acc = val_total_correct / val_total_samples
                        val_loss = val_total_loss / val_total_samples
                        pred_test = np.concatenate(all_preds, axis=0)
                        true_test = np.concatenate(all_labels, axis=0)
                        #pred_test = torch.cat(all_preds, dim=0).cpu().detach().numpy()
                        #true_test = torch.cat(all_labels, dim=0).cpu().detach().numpy()

                    # calculate score)
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