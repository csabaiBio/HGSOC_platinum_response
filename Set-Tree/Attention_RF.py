import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import settree
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import argparse


def gen_data(df,folder_path,target="brca"):
    # Dictionary to store features
    X = []
    y = []
    for index, row in tqdm(df.iterrows()):
        print(row['slide_id'])
        file_name = f"{row['slide_id']}.h5"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as h5_file:
                # Assuming the features are stored under a key named 'features'
                features = np.array(h5_file['features'])
                X.append(features)
                if target =="brca":
                    label = [1 if row["brca_status"] == "positive" else 0]  # this needs to be editable...
                if target =="HGSOC":
                    label = [1 if row["label"] == "sensitive" else 0]  # this needs to be editable...
                if target =="HunCRC":
                    if row["category"] == "negative":
                        label = [0]
                    elif row["category"] == "adenoma":
                        label = [1]
                    elif row["category"] == "non_neoplastic_lesion":
                        label = [2]
                    else:
                        label = [3]

                y.append(label[0])
        else:
            print(f"File not found: {file_name}")

    return X, y 


def train(ds_train,y_train):

    set_tree_model = settree.SetTree(classifier=True,
                                    criterion='entropy',
                                    splitter='sklearn',
                                    max_features=None,
                                    min_samples_split=2,
                                    operations=settree.OPERATIONS,
                                    use_attention_set=USE_ATTN_SET,
                                    use_attention_set_comp=USE_ATTN_SET_COMP,
                                    attention_set_limit=ATTN_SET_LIMIT,
                                    max_depth=MAX_DEPTH,
                                    min_samples_leaf=None,
                                    random_state=SEED)

    set_tree_model.fit(ds_train, np.array(y_train))

    return set_tree_model


def train_Grad_boost(ds_train,y_train):

    gbest_model = settree.GradientBoostedSetTreeClassifier(learning_rate=0.1, 
                                                       n_estimators=10,
                                                       criterion='mse',
                                                       operations=settree.OPERATIONS,
                                                       use_attention_set=True,
                                                       use_attention_set_comp=True,
                                                       attention_set_limit=2,
                                                       max_depth=3)
    gbest_model.fit(ds_train, y_train)

    return set_tree_model



# This condition checks if the script is run directly (not imported)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A simple script with command-line arguments")
    # Command-line arguments
    parser.add_argument('--train_csv', type=str, default="/mnt/ncshare/ozkilim/BRCA/data/tasks/crc.csv", help='csv for training set')
    parser.add_argument('--train_h5_path', type=str, default="/home/qbeer/GitHub/brca/data/hunCRC_CLAM/level_0/CLAM_ViT_features/h5_files/", help='embeddings of WSI patches in bags')
    parser.add_argument('--test_csv', type=str, default="/mnt/ncshare/ozkilim/BRCA/data/tasks/multiplexed_CRC.csv", help='csv for training set')
    parser.add_argument('--test_h5_path', type=str, default="/tank/WSI_data/CRC_multiplexed/CLAM/level_0/ViT/h5_files/", help='embeddings of WSI patches in bags')
    parser.add_argument('--task_type', type=str, default="test", help='if to run an internal split or use an external validation set')
    parser.add_argument('--train_target', type=str, default="HunCRC", help='task to run')
    parser.add_argument('--test_target', type=str, default="HunCRC", help='task to run')

    # Parse the command-line arguments
    args = parser.parse_args()
    # Model params
    ATTN_SET_LIMIT = 1
    USE_ATTN_SET = True
    USE_ATTN_SET_COMP = True
    MAX_DEPTH = 3
    SEED = 0

    train_df = pd.read_csv(args.train_csv, dtype={'slide_id': str})
    print("getting train data")
    X, y = gen_data(train_df,args.train_h5_path,args.train_target)

    print("splitting data")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
    print(y_train)
    test_df = pd.read_csv(args.test_csv, dtype={'slide_id': str})
    X_test, y_test = gen_data(test_df,args.test_h5_path,args.test_target)

    ds_train = settree.SetDataset(records=X_train, is_init=True)
    ds_val = settree.SetDataset(records=X_val, is_init=True)
    ds_test = settree.SetDataset(records=X_test, is_init=True)

    print("training model")
    set_tree_model = train(ds_train,y_train)
    print("internal val")
    preds = set_tree_model.predict(ds_val)

    print(preds)
    print(y_val)

    y_val = np.array(y_val)

    auc = roc_auc_score(y_val, preds) #why 1?...

    print('Set-Tree internal val : AUC: {:.4f}'.format(auc))

    if args.task_type == "test":
        print("external test")
        preds = set_tree_model.predict_proba(ds_test)
        print(preds)
        # auc = roc_auc_score(y_test, preds)
        # print('Set-Tree external test : AUC: {:.4f}'.format(auc))


    ## save attentions for further downstream analysis.
    sample_record = X_test[0]
    point2rank = settree.get_item2rank_from_tree(set_tree_model, settree.SetDataset(records=[sample_record], is_init=True))
    attnetions = np.array(list(point2rank.values()))
    
    np.save("CRC_attentions_RF.npy",attnetions)


# [1, 3, 0, 3, 1, 3, 1, 1, 1, 2, 2, 3, 3, 1, 2, 1, 3, 3, 1, 3, 2, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 0, 3, 2]


# [3, 3, 2, 2, 1, 3, 1, 1, 1, 2, 3, 3, 3, 0, 2, 2, 2, 3, 1, 3, 2, 1, 3, 3, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 3, 2]