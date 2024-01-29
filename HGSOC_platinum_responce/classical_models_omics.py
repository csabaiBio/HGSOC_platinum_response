# load splits and run all clasical experiments and load in pickles for plotting in tables.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import os 
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.ensemble import VotingClassifier
import json



def get_rf_auc_list(split_folder_path,prots):
    '''Perform classical experiments with omics data only as in Chowdry et al'''
    test_aucs = []
    # Loop over split files and create train, val, test subsets
    for filename in os.listdir(split_folder_path):
        if filename.endswith('.csv'):  # Assuming the files are in CSV format use given format... 
            
            split_df = pd.read_csv(os.path.join(split_folder_path, filename))
            # Extract identifiers
        
            train_df = df[df['slide_id'].isin(split_df["train"].to_list())]
            val_df = df[df['slide_id'].isin(split_df["val"].to_list())]
            test_df = df[df['slide_id'].isin(split_df["test"].to_list())]

            X_train = train_df[prots]  # Replace 'label_column' with your label column name
            y_train = train_df['label']  # Replace 'label_column' with your label column name

            X_test = test_df[prots]  # Replace 'label_column' with your label column name
            y_test = test_df['label']  # Replace 'label_column' with your label column name

            # Initialize and train the Random Forest classifier

            random_forest = RandomForestRegressor()
            elastic_net = ElasticNet()
            xgboost_model = XGBRegressor()

            ensemble = VotingRegressor(
            estimators=[
                    ('random_forest', random_forest),
                    ('elastic_net', elastic_net),
                    ('xgboost', xgboost_model)
                ]
            )

            # rf_clf = RandomForestClassifier()
            ensemble.fit(X_train, y_train)

            # Make predictions on the test set
            rf_pred = ensemble.predict(X_test)

            # Evaluate the Random Forest model
            # rf_accuracy = accuracy_score(y_test, rf_pred)
            rf_roc_auc = roc_auc_score(y_test, rf_pred)
            # print(f"Random Forest ROC AUC: {rf_roc_auc}")
            test_aucs.append(rf_roc_auc)
            
    return test_aucs



df = pd.read_csv("HGSOC_TCGA_main.csv",header=0,low_memory=False)
# provied 60 protein signature.
prots_60 = ['RAB25', 'BCL2L1', 'HADH', 'NFKB2', 'COX7A2', 'COX7C', 'TPMT', 'GOLPH3L', 'LTA4H', 'COX6C', 'IDH1', 'YWHAG', 'S100A10', 'COX6A1', 'NDUFB3', 'TGM2', 'CDKN1B', 'NFKB1', 'CAMK2D', 'IL4I1', 'FDX1', 'VCAM1', 'ATM', 'NCAPH2', 'ABCB8', 'IDI1', 'PLIN2', 'ATP6V1D', 'GPX4', 'CA2', 'RELA', 'GLUD1', 'TOP3B', 'RPS6KB2', 'KEAP1', 'LGALS1', 'MTDH', 'AIFM1', 'RHOA', 'CASP7', 'PTGES2', 'TFRC', 'CHUK', 'GPX1', 'PDK1', 'STAT3', 'PECR', 'TALDO1', 'XIAP', 'ACADSB', 'CPOX', 'ARNT', 'BIRC2', 'ACOT7', 'HACL1', 'MYD88', 'EGFR', 'RIPK1', 'NBN', 'LDHA']

with open('proteomics_combinations.json', 'r') as file:
			protein_sets = json.load(file)

prots_1k = protein_sets['TCGA_flat_1k']

# Part 1. HGSOC train TCGA test
# Part 2. TCGA train HGSOC test
# Part 3. HGSOC hospital hold out splits

splits_folder = 'splits'
classical_results_60 = {}
classical_results_1k = {}

# Loop over each directory and subdirectory in the main folder
for root, subfolders, files in os.walk(splits_folder):
    for subfolder in subfolders:
        # Construct the path to the subfolder
        split_folder_path = os.path.join(root, subfolder)

        aucs_60 = get_rf_auc_list(split_folder_path,prots_60)
        aucs_1k = get_rf_auc_list(split_folder_path,prots_1k)

        classical_results_60[split_folder_path.split('/')[-1]]=aucs_60
        classical_results_1k[split_folder_path.split('/')[-1]]=aucs_1k

# Save results for paper tables.
classical_results = [classical_results_60,classical_results_1k]

with open('classical_results_stratified.json', 'w') as json_file:
    json.dump(classical_results, json_file)

