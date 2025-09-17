import numpy as np
import pandas as pd
import os
import pickle

# loading raw features data
path_dataset = "before MCI features.pickle"

k = 3
model = 'Magnet'
# root_path = "/blue/bianjiang/fanz/AD_results_20241107/nebor_200/Results_gamma_0_dim_64/Output/Cluster/"


# loading files
file_dataset = open(path_dataset, 'rb')
df_features = pickle.load(file_dataset)

# List of columns to drop
columns_to_drop = [ 'label', 'status', 'next_status', 'outcome_final_AD', '(0.0, 49.0]']#'PATID','ADMIT_DATE',
# Drop the columns if they exist
raw_data = df_features.drop(columns=columns_to_drop, errors='ignore')

"""
cohort_df = pd.read_csv(root_dir + processed_dir + "/mci_ad_study_cohort_with_last_MCI.csv")[['ID','first_MCI_date']].rename(columns={'ID': 'PATID'})
raw_data = raw_data.merge(cohort_df, on='PATID', how='left')
raw_data = raw_data[raw_data['ADMIT_DATE']<raw_data['first_MCI_date']]
raw_data = raw_data.drop(columns=['PATID','ADMIT_DATE','first_MCI_date'], errors='ignore')
"""

# Create a mapping of old names to new names
problematic_features = [
    col for col in df_features.columns 
    if any(char in col for char in ['[', ']', '<']) and not col.startswith('AGE')
]
rename_dict = {col: col.replace('[', '').replace(']', '').replace('<', '') for col in problematic_features}
raw_data.rename(columns=rename_dict, inplace=True)

raw_data.reset_index(inplace=True)


# Define the function to rename columns based on the provided conditions.
def rename_columns(df):
    # Create a dictionary for the new column names.
    new_column_names = {}
    # Iterate through each column in the dataframe.
    for column in df.columns:
        if column.startswith('Phe_'):
            new_column_names[column] = column.replace('Phe_', '')
        elif column.startswith('ATC_'):
            new_column_names[column] = column.replace('ATC_', '').lower().capitalize()
        elif column == 'F':
            new_column_names[column] = 'Gender_Female'
        elif column == 'M':
            new_column_names[column] = 'Gender_Male'

    # Rename the columns in the dataframe.
    df.rename(columns=new_column_names, inplace=True)

    return df


# Apply the function to rename columns
raw_data = rename_columns(raw_data)

"""
# Assuming 'Race' is already in df_features as integers 0, 1, 2, 3
# Create one-hot encoded columns
race_dummies = pd.get_dummies(df_race['Race'], prefix='Race')

# Rename the columns to the corresponding race categories
race_dummies.columns = ['Race_NHB', 'Race_NHW', 'Race_OT', 'Race_UN']
race_dummies
"""

dataset = 'whole_dataset'
file_name = f"Cluster_k{k}.csv"
before_mci_file_name = f'Cluster_before_MCI_k{k}.csv'


cluster_path = f'../S2.clustering/Output/Cluster/{model}/whole_dataset/Cluster_k{k}.csv'
cluster_data = pd.read_csv(cluster_path)[['ENCID', 'cluster_info']]

before_mci_cluster_path =f'../S2.clustering/Output/Cluster/{model}/whole_dataset/Cluster_before_MCI_k{k}.csv'
before_mci_cluster_data = pd.read_csv(before_mci_cluster_path)[['ENCID', 'cluster_info']]

data = raw_data.merge(before_mci_cluster_data, on='ENCID', how='left').rename(columns={'cluster_info': f'Current_Subphonetype'})
data = data.merge(cluster_data, on='ENCID', how='left').rename(columns={'cluster_info': f'{model}_cluster'})
data = data.drop(columns=['ENCID','current_status', 'ADMIT_DATE'], errors='ignore')

# Define which columns need the 'first' aggregation
first_columns = []
# All other columns will use 'max' aggregation
max_columns = [col for col in data.columns if col not in first_columns + ['PATID']]
# Define the aggregation dictionary
aggregations = {col: 'max' for col in max_columns}
aggregations.update({col: 'first' for col in first_columns})
# Perform the groupby and aggregation
data = data.groupby('PATID').agg(aggregations).reset_index()
print(data)

data.to_csv(f'{model} k={k} data.csv', index = False)



"""
# Assuming 'data' is your DataFrame and it has been loaded already
selected_columns = ['GAT_cluster','GCN_cluster', 'GraphSAGE_cluster', 'Magnet_cluster']

# Compute the correlation only for the selected columns with all other columns
# Create an empty DataFrame to store correlations
correlation_matrix = pd.DataFrame(index=data.columns)

# Loop through each of the selected columns and calculate its correlation with all others
for column in selected_columns:
    # Compute correlations with other columns and assign to the empty DataFrame
    correlation_matrix[column] = data.corrwith(data[column])
correlation_matrix.to_csv('whole_data_correlation_matrix_1201.csv')
"""
