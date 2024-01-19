# script to create tasks with args 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


status = pd.read_csv('/mnt/ncshare/ozkilim/BRCA/data/TCGA_metadata/TCGA_OV_HRDstatus.txt', sep='\t')
filtered_status = status[['PatientID', 'HRDetect', 'BRCA-status']].copy()


paths = list(Path('/tank/WSI_data/Ovarian_WSIs/TCGA-OV').glob('*.svs'))


slide_info = np.array([['-'.join(path.name.split('-')[:3]), path.name.split('.')[0][-3:]] for path in paths])
slide_names, slide_types = slide_info[:, 0], slide_info[:, 1]


# NOTE: update the dataframe with the slide names corresponding to the status file
filtered_status = status[['PatientID', 'HRDetect', 'BRCA-status']].copy()

filtered_status['slide_paths'] = [None] * len(filtered_status)
filtered_status['slide_types'] = [None] * len(filtered_status)


for ind, row in filtered_status.iterrows():
    filtered_status.loc[ind, 'slide_paths'] = ','.join([str(path).split('/')[-1].replace('.svs', '') for path in paths if row['PatientID'] in str(path)])
    filtered_status.loc[ind, 'slide_types'] = ','.join([slide_type for name, slide_type in zip(slide_names, slide_types) if row['PatientID'] in name])


dataset = []
for _, row in filtered_status.iterrows():
    for slide_id, slide_type in zip(row['slide_paths'].split(','), row['slide_types'].split(',')):
        if row['BRCA-status'] == 'BRCA1_deficient' or row['BRCA-status'] == 'BRCA2_deficient':
                brca_status = 'positive'
        elif row['BRCA-status'] == 'quiescent':
                    brca_status = 'negative'
        elif row['BRCA-status'] == 'intact':
                    brca_status = 'negative'
        sample = {
            'case_id': row['PatientID'],
            'slide_id': slide_id,
            'slide_type': slide_type,
            'brca_status': brca_status,
            'hrd_score': row['HRDetect']
        }

        dataset.append(sample)

dataset = pd.DataFrame(dataset)
dataset = dataset.sample(frac=1, random_state=137).reset_index(drop=True)

print(dataset.head())




# Fuze in responce data..

# Fuse all tcga into a master DF. each row is a slide....

# Then can train on all lables on interest?........


# threshold = 0.7
# # Create a new column based on binary thresholding
# dataset['hrd_status'] = dataset['hrd_score'].apply(lambda x: 1 if x >= threshold else 0)
# df_hrd_status = dataset[["case_id","slide_id","hrd_status"]]
# df_hrd_status.to_csv("../data/raw_subsets/HRD_binary.csv",index=False)

# #binary split no intact slides
# pos_neg = dataset[dataset["brca_status"] != "intact"]
# pos_neg.to_csv('../data/brca_dataset_pos_neg.csv', index=False)

# FFPE only
# FFPE_only = dataset[dataset['slide_type'].isin(["DX1","DX2"])]
# FFPE_only.to_csv('../data/brca_dataset_FFPE_NERO.csv', index=False)

# # TS only 
# TS_only = dataset[dataset['slide_type'].isin(["TS1","TS2"])] #
# TS_only.to_csv('../data/brca_dataset_TS_1_2.csv', index=False)


# # binary split and FFPE..
# FFPE_only_binary = pos_neg[pos_neg['slide_type'].isin(["DX1","DX2"])]
# FFPE_only_binary.to_csv('../data/brca_dataset_FFPE_binary.csv', index=False)