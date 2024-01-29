from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pylab as plt
import glob
import os
import numpy as np
import pandas as pd 
from scipy import stats
import json
from sklearn.metrics import roc_auc_score
import re


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

            joined_dir = "/mnt/ncshare/ozkilim/BRCA/results/platinum_responce_results/" + root_dir
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


def get_ensemble_auc_scores_from_files(root_dirs):
    '''Load up lost of root dirs actualy preds and make ensemble before returning ensemble AUC'''
    # loop over each split pred set. 
    all_aucs = []
    for i in range(5):
        accumulated_prob = None
        labels = None
        for root_dir in root_dirs: 
            file = "../../results/platinum_responce_results" + root_dir + "/split_"+ str(i) +"_results.pkl"
            #read the pkl 
            df = pd.read_pickle(file)
            data = []
            for key, value in df.items():
                slide_id = value['slide_id']
                prob = value['prob'][0][1]
                label = value['label']
                data.append({'slide_id': slide_id, 'prob': prob, 'label': label})
            # Creating DataFrame
            df = pd.DataFrame(data)

            # Check and store the 'label' values
            if labels is None:
                labels = df['label']
            elif not df['label'].equals(labels):
                raise ValueError("Inconsistent 'label' values across dataframes")
            # Accumulate 'prob' values
            if accumulated_prob is None:
                accumulated_prob = df['prob']
            else:
                accumulated_prob += df['prob']
                
        # Average the accumulated probabilities
        ensemble_prob = accumulated_prob / len(root_dirs)
        # Calculate AUC
        auc = roc_auc_score(labels, ensemble_prob)
        all_aucs.append(auc)
    
    return all_aucs



def process_auc_files(results_dict):
    # Initialize an empty DataFrame to store all AUC scores
    # Loop through each file and extract AUC scores
    all_auc_scores = []
    # loop overdict...  

    items_list = list(results_dict)
    # Loop over the dictionary by index
    for i in range(len(items_list)):

        if "ENSEMBLE" in list(results_dict[i].keys())[0]:

            get_ensemble_auc_scores_from_files(list(results_dict[i].values())[0][0])
        
            category = list(results_dict[i].values())[0][1]
            embedder = list(results_dict[i].values())[0][2]

            mean_auc = round(np.mean(aucs),3)
            std = round(np.std(aucs),3)
        else:
                    
            aucs = get_auc_scores_from_file(list(results_dict[i].values())[0][0]) #always spulls same key
            category = list(results_dict[i].values())[0][1]
            embedder = list(results_dict[i].values())[0][2]

            # get stats here... 
            mean_auc = round(np.mean(aucs),3)
            std = round(np.std(aucs),3)

        row = {"model":list(results_dict[i].keys())[0], "TCGA":mean_auc.astype(str)+"±"+std.astype(str), "category":category,"embedder":embedder}

        all_auc_scores.append(row)
    
    all_auc_scores = pd.DataFrame(all_auc_scores)

    # Sort the DataFrame based on 'Mean AUC' in descending order
    sorted_auc_summary = all_auc_scores.sort_values(by='TCGA', ascending=True)
    
    return sorted_auc_summary


def format_highest_values(df):
    for col in df.columns[2:]:
        highest_value = df[col].max()
        df[col] = df[col].apply(lambda x: f'\\textbf{{{x}}}' if x == highest_value else x)
    return df



def parse_value(value):
    # Function to parse the numerical part before '±'
    if pd.notna(value):
        match = re.search(r"(\d+\.\d+)(?=±)", value)
        if match:
            return float(match.group(1))
    return np.nan



def generate_latex_table(df_pivot, header, caption):
    # Create a DataFrame to store the parsed numeric values
    df_numeric = pd.DataFrame()

    # Apply parsing to each relevant column (considering columns from index 2 onwards)
    data_columns = df_pivot.columns[2:]
    for col in data_columns:
        df_numeric[col] = df_pivot[col].apply(parse_value)

    # Identify the largest and second largest value in each numeric column
    max_values = df_numeric.max()
    second_max_values = df_numeric.apply(lambda x: x.nlargest(2).iloc[-1] if x.nlargest(2).size > 1 else np.nan)

    # Start the table and add the header
    latex_str = "\\begin{table}[ht]\n\\footnotesize\n\\centering\n\\begin{tabular}{cc|ccc|ccc}\n\\toprule\n"
    latex_str += " & \\multicolumn{1}{c}{" + header + "} & \\multicolumn{3}{c}{Primary} & \\multicolumn{3}{c}{Metastatic} \\\\\n"
    latex_str += "\\midrule\n"
    latex_str += " & Model & CTransPath \\cite{wang2022transformer} & Lunit-Dino \\cite{kang2023benchmarking} & OV-Dino (ours) & CTransPath & Lunit-Dino & OV-Dino \\\\\n"
    latex_str += "\\midrule\n"

    # Add rows from the DataFrame
    for category, group_df in df_pivot.groupby('category'):
        group_len = len(group_df)
        latex_str += f"\\multirow{{{group_len}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\tiny {category}}}}} \n"
        for _, row in group_df.iterrows():
            model = row['model']
            values = []
            for col in data_columns:
                value = row[col]
                # Check if the value is the maximum or second maximum in its column
                if df_numeric.at[_, col] == max_values[col]:
                    value = f"\\textbf{{{value}}}"
                elif df_numeric.at[_, col] == second_max_values[col]:
                    value = f"\\underline{{{value}}}"
                values.append(value)
            values_str = ' & '.join([str(model)] + values)
            latex_str += f" & {values_str} \\\\\n"
        latex_str += "\\midrule\n"

    latex_str += "\\bottomrule\n\\end{tabular}\n\\vspace{6pt}\n\\caption{" + caption + "}\n\\end{table}"

    return latex_str


def find_value(s):
    if pd.notna(s):
        match = re.search(r'\b\d+\.?\d*±\d+\.?\d*\b', s)
        if match:
            return match.group(0)
    return np.nan

######## main #########

# need baseline omics scroes.
with open('classical_results.json', 'r') as json_file:
    classical_results = json.load(json_file)


# Part 1. HGSOC train TCGA test
dfs = []
for tissue_type in ["primary","metastatic"]:
    # factorize to load all tables from runs.. more automated less room for bugs... 
    results_dict = [{"60 protein ensemble \cite{chowdhury2023proteogenomic}":[classical_results[0]['HGSOC_train_TCGA_test_' + tissue_type.capitalize()],"Omics","ViT"]}, {"1k protein ensemble":[classical_results[1]['HGSOC_train_TCGA_test_' + tissue_type.capitalize()],"Omics","ViT"]}]
    embeddings = ["ViT","OV_ViT","CTransPath"]

    for embedding in embeddings: 

        results_dict.append({"clam\_sb \cite{lu2021data}":["HGSOC_TRAIN_TCGA_15_clam_sb_None_"+embedding+"_"+tissue_type+"_s1","WSI",embedding]})

        results_dict.append({"PorpoiseMMF 60 \cite{chen2022pan}":["HGSOC_TRAIN_TCGA_15_PorpoiseMMF_concat_60_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"PorpoiseMMF 1k \cite{chen2022pan}":["HGSOC_TRAIN_TCGA_15_PorpoiseMMF_concat_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

        results_dict.append({"MCAT 60 \cite{chen2021multimodal}":["HGSOC_TRAIN_TCGA_15_MCAT_Surv_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"MCAT 1k \cite{chen2021multimodal}":["HGSOC_TRAIN_TCGA_15_MCAT_Surv_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"MCAT plat_resp \cite{chen2021multimodal}":["HGSOC_TRAIN_TCGA_15_MCAT_Surv_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})


        results_dict.append({"SurvPath 60 \cite{jaume2023modeling}":["HGSOC_TRAIN_TCGA_15_SurvPath_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"SurvPath 1k \cite{jaume2023modeling}":["HGSOC_TRAIN_TCGA_15_SurvPath_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"SurvPath plat_resp \cite{jaume2023modeling}":["HGSOC_TRAIN_TCGA_15_SurvPath_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

    df = process_auc_files(results_dict)
    # print(df.head(20))
    # Pivot the DataFrame
    df_pivot = df.pivot_table(index=['model', 'category'], columns='embedder', values=['TCGA'], aggfunc='first').reset_index()
    df_pivot.columns = [' '.join(col).strip() for col in df_pivot.columns.values]
    
    # Populate nans for classical models...
    for index, row in df_pivot.iterrows():
        values = row.apply(find_value).dropna()
        if not values.empty:
            value_to_fill = values.values[0]
            df_pivot.loc[index] = df_pivot.loc[index].fillna(value_to_fill)

    dfs.append(df_pivot)


# Merge the DataFrames
merged_df = dfs[0].merge(dfs[1], on='model', suffixes=('_primary', '_metastatic'))
# Rename the 'name' column after merging
merged_df.drop(['category_primary'], axis=1, inplace=True)
merged_df.rename(columns={'category_metastatic': 'category'}, inplace=True)
# Get the column you want to move
column_to_move = merged_df.pop('category')
# Insert the column at the desired position (position 2)
merged_df.insert(1, 'category', column_to_move)
print(merged_df.head())

# Generate LaTeX table
header = "HGSOC train TCGA test"
caption = "Testing on TCGA samples \cite{cancer2011integrated} AUC scores. All primary tumor samples from the discovery dataset are used for training. Bold values are the highest scores for a given feature extractor and architecture. Underlined are the second-highest scores."

latex_table = generate_latex_table(merged_df,header,caption)
# save table to tex file with header name of part/experiment
print(latex_table)
# find way to combine primary and metrastatics to one df ... for saving in full... 

with open("results_tables/HGSOC_TRAIN_TCGA_TEST.tex", 'w') as f:
    f.write(latex_table)





# Part 2. TCGA train HGSOC test
# factorize to load all tables from runs.. more automated less room for bugs... 
dfs = []
for tissue_type in ["primary","metastatic"]:
    results_dict = [{"60 protein ensemble \cite{chowdhury2023proteogenomic}":[classical_results[0]['TCGA_train_HGSOC_test_' + tissue_type.capitalize()],"Omics","ViT"]},{"1k protein ensemble":[classical_results[1]['TCGA_train_HGSOC_test_' + tissue_type.capitalize()],"Omics","ViT"]}]
    embeddings = ["ViT","OV_ViT","CTransPath"]

    for embedding in embeddings: 

        results_dict.append({"clam\_sb \cite{lu2021data}":["TCGA_TRAIN_HGSOC_15_clam_sb_None_"+embedding+"_"+tissue_type+"_s1","WSI",embedding]})

        results_dict.append({"PorpoiseMMF 60 \cite{lu2021data}":["TCGA_TRAIN_HGSOC_15_PorpoiseMMF_concat_60_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"PorpoiseMMF 1k \cite{lu2021data}":["TCGA_TRAIN_HGSOC_15_PorpoiseMMF_concat_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

        results_dict.append({"MCAT 60 \cite{lu2021data}":["TCGA_TRAIN_HGSOC_15_MCAT_Surv_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"MCAT 1k \cite{lu2021data}":["TCGA_TRAIN_HGSOC_15_MCAT_Surv_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"MCAT plat_resp \cite{lu2021data}":["TCGA_TRAIN_HGSOC_15_MCAT_Surv_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

        results_dict.append({"SurvPath 60 \cite{lu2021data}":["TCGA_TRAIN_HGSOC_15_SurvPath_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"SurvPath 1k \cite{lu2021data}":["TCGA_TRAIN_HGSOC_15_SurvPath_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"SurvPath plat_resp \cite{lu2021data}":["TCGA_TRAIN_HGSOC_15_SurvPath_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})


    df = process_auc_files(results_dict)
    # print(df.head(20))
    # Pivot the DataFrame
    df_pivot = df.pivot_table(index=['model', 'category'], columns='embedder', values=['TCGA'], aggfunc='first').reset_index()
    df_pivot.columns = [' '.join(col).strip() for col in df_pivot.columns.values]
    print(df_pivot.head(20))

    # Populate nans for classical models...
    for index, row in df_pivot.iterrows():
        values = row.apply(find_value).dropna()
        if not values.empty:
            value_to_fill = values.values[0]
            df_pivot.loc[index] = df_pivot.loc[index].fillna(value_to_fill)

    dfs.append(df_pivot)


# Merge the DataFrames
merged_df = dfs[0].merge(dfs[1], on='model', suffixes=('_primary', '_metastatic'))
# Rename the 'name' column after merging
merged_df.drop(['category_primary'], axis=1, inplace=True)
merged_df.rename(columns={'category_metastatic': 'category'}, inplace=True)
# Get the column you want to move
column_to_move = merged_df.pop('category')
# Insert the column at the desired position (position 2)
merged_df.insert(1, 'category', column_to_move)
print(merged_df.head())

# Generate LaTeX table
header = "TCGA train HGSOC test"
caption = "Testing on HGSOC samples AUC scores. All primary tumor samples from the discovery dataset are used for training. Bold values are the highest scores for a given feature extractor and architecture. Underlined are the second-highest scores."

latex_table = generate_latex_table(merged_df,header,caption)

# save table to tex file with header name of part/experiment
print(latex_table)
# save table
with open("results_tables/TCGA_TRAIN_HGSOC_TEST_.tex", 'w') as f:
    f.write(latex_table)




# Part 3. HGSOC hospital hold out splits
# factorize to load all tables from runs.. more automated less room for bugs... 
    
dfs = []
for tissue_type in ["primary","metastatic"]:

    results_dict = [{"60 protein ensemble \cite{chowdhury2023proteogenomic}":[classical_results[0]['HGSOC_UAB_hold_out_' + tissue_type.capitalize()],"Omics","ViT"]},{"1k protein ensemble":[classical_results[1]['HGSOC_UAB_hold_out_' + tissue_type.capitalize()],"Omics","ViT"]}]
    embeddings = ["ViT","OV_ViT","CTransPath"]

    for embedding in embeddings: 

        results_dict.append({"clam\_sb \cite{lu2021data}":["HGSOC_UAB_hold_out_15_clam_sb_None_"+embedding+"_"+tissue_type+"_s1","WSI",embedding]})

        results_dict.append({"PorpoiseMMF 60 \cite{lu2021data}":["HGSOC_UAB_hold_out_15_PorpoiseMMF_concat_60_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"PorpoiseMMF 1k \cite{lu2021data}":["HGSOC_UAB_hold_out_15_PorpoiseMMF_concat_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

        results_dict.append({"MCAT 60 \cite{lu2021data}":["HGSOC_UAB_hold_out_15_MCAT_Surv_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"MCAT 1k \cite{lu2021data}":["HGSOC_UAB_hold_out_15_MCAT_Surv_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"MCAT plat_resp \cite{lu2021data}":["HGSOC_UAB_hold_out_15_MCAT_Surv_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

        results_dict.append({"SurvPath 60 \cite{lu2021data}":["HGSOC_UAB_hold_out_15_SurvPath_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"SurvPath 1k \cite{lu2021data}":["HGSOC_UAB_hold_out_15_SurvPath_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"SurvPath plat_resp \cite{lu2021data}":["HGSOC_UAB_hold_out_15_SurvPath_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

    df = process_auc_files(results_dict)
    # print(df.head(20))
    # Pivot the DataFrame
    df_pivot = df.pivot_table(index=['model', 'category'], columns='embedder', values=['TCGA'], aggfunc='first').reset_index()
    df_pivot.columns = [' '.join(col).strip() for col in df_pivot.columns.values]
    print(df_pivot.head(20))

    for index, row in df_pivot.iterrows():
        values = row.apply(find_value).dropna()
        if not values.empty:
            value_to_fill = values.values[0]
            df_pivot.loc[index] = df_pivot.loc[index].fillna(value_to_fill)

    dfs.append(df_pivot)


# Merge the DataFrames
merged_df = dfs[0].merge(dfs[1], on='model', suffixes=('_primary', '_metastatic'))
# Rename the 'name' column after merging
merged_df.drop(['category_primary'], axis=1, inplace=True)
merged_df.rename(columns={'category_metastatic': 'category'}, inplace=True)
# Get the column you want to move
column_to_move = merged_df.pop('category')
# Insert the column at the desired position (position 2)
merged_df.insert(1, 'category', column_to_move)
print(merged_df.head())

# Generate LaTeX table
header = "HGSOC_UAB_hold_out"
caption = "HGSOC_UAB_hold_out. Bold values are the highest scores for a given feature extractor and architecture. Underlined are the second-highest scores."

latex_table = generate_latex_table(merged_df,header,caption)

# save table to tex file with header name of part/experiment
print(latex_table)
with open("results_tables/HGSOC_UAB_hold_out.tex", 'w') as f:
    f.write(latex_table)




# factorize to load all tables from runs.. more automated less room for bugs... 
dfs = []
for tissue_type in ["primary","metastatic"]:

    results_dict = [{"60 protein ensemble \cite{chowdhury2023proteogenomic}":[classical_results[0]['HGSOC_MAYO_hold_out_' + tissue_type.capitalize()],"Omics","ViT"]}, {"1k protein ensemble":[classical_results[1]['HGSOC_MAYO_hold_out_' + tissue_type.capitalize()],"Omics","ViT"]}]
    embeddings = ["ViT","OV_ViT","CTransPath"]

    for embedding in embeddings: 

        results_dict.append({"clam\_sb \cite{lu2021data}":["HGSOC_MAYO_hold_out_15_clam_sb_None_"+embedding+"_"+tissue_type+"_s1","WSI",embedding]})

        results_dict.append({"PorpoiseMMF 60 \cite{lu2021data}":["HGSOC_MAYO_hold_out_15_PorpoiseMMF_concat_60_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"PorpoiseMMF 1k \cite{lu2021data}":["HGSOC_MAYO_hold_out_15_PorpoiseMMF_concat_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

        results_dict.append({"MCAT 60 \cite{lu2021data}":["HGSOC_MAYO_hold_out_15_MCAT_Surv_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"MCAT 1k \cite{lu2021data}":["HGSOC_MAYO_hold_out_15_MCAT_Surv_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"MCAT plat_resp \cite{lu2021data}":["HGSOC_MAYO_hold_out_15_MCAT_Surv_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

        results_dict.append({"SurvPath 60 \cite{lu2021data}":["HGSOC_MAYO_hold_out_15_SurvPath_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"SurvPath 1k \cite{lu2021data}":["HGSOC_MAYO_hold_out_15_SurvPath_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
        results_dict.append({"SurvPath plat_resp \cite{lu2021data}":["HGSOC_MAYO_hold_out_15_SurvPath_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

    df = process_auc_files(results_dict)
    # print(df.head(20))
    # Pivot the DataFrame
    df_pivot = df.pivot_table(index=['model', 'category'], columns='embedder', values=['TCGA'], aggfunc='first').reset_index()
    df_pivot.columns = [' '.join(col).strip() for col in df_pivot.columns.values]
    print(df_pivot.head(20))

    for index, row in df_pivot.iterrows():
        values = row.apply(find_value).dropna()
        if not values.empty:
            value_to_fill = values.values[0]
            df_pivot.loc[index] = df_pivot.loc[index].fillna(value_to_fill)

    dfs.append(df_pivot)


# Merge the DataFrames
merged_df = dfs[0].merge(dfs[1], on='model', suffixes=('_primary', '_metastatic'))
# Rename the 'name' column after merging
merged_df.drop(['category_primary'], axis=1, inplace=True)
merged_df.rename(columns={'category_metastatic': 'category'}, inplace=True)
# Get the column you want to move
column_to_move = merged_df.pop('category')
# Insert the column at the desired position (position 2)
merged_df.insert(1, 'category', column_to_move)
print(merged_df.head())

# Generate LaTeX table
header = "HGSOC_MAYO_hold_out"
caption = "HGSOC_MAYO_hold_out. Bold values are the highest scores for a given feature extractor and architecture. Underlined are the second-highest scores."

latex_table = generate_latex_table(merged_df,header,caption)

print(latex_table)
with open("results_tables/HGSOC_MAYO_hold_out.tex", 'w') as f:
    f.write(latex_table)