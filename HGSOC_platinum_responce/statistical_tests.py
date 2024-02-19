from scipy import stats
from tensorflow.python.summary.summary_iterator import summary_iterator
import os
import numpy as np
import pandas as pd 
import glob
import json 



def test_auc(tf_path):
    """Take a tf output path and create a list of values for the validtion auc of an experement"""
    auc = []
    for event in summary_iterator(tf_path):
        for value in event.summary.value:
            if value.tag == "final/test_auc":
                auc.append(value.simple_value)
    return auc

def get_auc_scores_from_file(root_dir):
    try:
        if isinstance(root_dir, str):
            all_aucs = []
            file_extension = "*.gpu1"

            joined_dir = "/mnt/ncshare/ozkilim/BRCA/results/platinum_responce_results_stratified/" + root_dir
            # Recursively traverse the directory and its subdirectories
            for subdir, _, files in os.walk(joined_dir):
                # Use glob to find all files with the specified extension
                for file in glob.glob(os.path.join(subdir, file_extension)):
                    auc = test_auc(file)
                    all_aucs.append(auc[0])
        else:
            all_aucs = root_dir.copy()
        
    except:
        all_aucs =  np.array([0,0,0,0,0,0,0,0,0])

    return all_aucs



def t_test(classical,root_dir_multimodal):
    """take the largest value from multimodal models and check significance relative to the best unimodal model. perfrom this for each column."""
    best_multimodal = get_auc_scores_from_file(root_dir_multimodal)
    # Perform a paired t-test
    print(classical)
    print(np.mean(classical))
    print(best_multimodal)
    print(np.mean(best_multimodal))

    t_statistic, p_value = stats.ttest_rel(best_multimodal,classical)

    print(f"t={t_statistic:.3} and p={p_value:.3}")



with open('classical_results.json', 'r') as json_file:
    classical_results = json.load(json_file)


classical = classical_results[0]['TCGA_train_HGSOC_test_Primary']
# given two models run rests to show if one is signifiactly better than another 
root_dir_multimodal = "TCGA_TRAIN_HGSOC_15_PorpoiseMMF_concat_1k_ViT_primary_s1"

t_test(classical,root_dir_multimodal)
