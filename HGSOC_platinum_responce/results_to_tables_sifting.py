from tensorflow.python.summary.summary_iterator import summary_iterator
import matplotlib.pylab as plt
import glob
import os
import numpy as np
import pandas as pd 
import json
from sklearn.metrics import roc_auc_score
import re


def test_auc(tf_path):
    """Take a tf output path and create a list of values for the validtion auc of an experement"""
    auc = []
    for event in summary_iterator(tf_path):
        for value in event.summary.value:
            if value.tag == "final/"+eval+"_auc":
                auc.append(value.simple_value)
    return auc

def get_auc_scores_from_file(root_dir):
    try:
        if isinstance(root_dir, str):
            all_aucs = []
            file_extension = "*.gpu1"

            joined_dir = "/mnt/ncshare/ozkilim/BRCA/results/platinum_responce_results_10"+lr+"/" + root_dir
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
            file = "/mnt/ncshare/ozkilim/BRCA/results/platinum_responce_results_10"+lr+"/" + root_dir + "/split_"+ str(i) +"_results.pkl"
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

        if list(results_dict[i].values())[0][2] =="ensemble":

            aucs = get_ensemble_auc_scores_from_files(list(results_dict[i].values())[0][0])
        
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
    latex_str = "\\begin{table}[ht]\n\\footnotesize\n\\centering\n\\begin{tabular}{cc|cccc|cccc}\n\\toprule\n"
    latex_str += " & \\multicolumn{1}{c}{" + header + "} & \\multicolumn{3}{c}{Primary} & \\multicolumn{3}{c}{Metastatic} \\\\\n"
    latex_str += "\\midrule\n"
    latex_str += " & Model &  Lunit-Dino \\cite{kang2023benchmarking} & OV-Dino (ours) &  CTransPath \\cite{wang2022transformer}  & ensemble & Lunit-Dino & OV-Dino &  CTransPath & ensemble \\\\\n"
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

    latex_str += "\\bottomrule\n\\end{tabular}\n\\vspace{6pt}\n\\caption{" + caption + "}\n\\label{tab:"+header+"}\\end{table}"

    return latex_str


def find_value(s):
    if pd.notna(s):
        match = re.search(r'\b\d+\.?\d*±\d+\.?\d*\b', s)
        if match:
            return match.group(0)
    return np.nan

def extract_numeric_values(value):
    """Extracts the numeric value and standard deviation from a string in the format 'value±std'."""
    match = re.match(r"([0-9.]+)±([0-9.]+)", value)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        return np.nan, np.nan  # Return NaN if format does not match
    
def calculate_mean_std(row, cols):
    values = [extract_numeric_values(row[col]) for col in cols]
    means = [val[0] for val in values if not np.isnan(val[0])]
    # stds = [val[1] for val in values if not np.isnan(val[1])]  # If you want to consider the original stds
    
    # Calculate mean and standard deviation of the means
    mean_of_means = np.mean(means)
    std_of_means = np.std(means)  # Standard deviation of the values
    
    return f"{mean_of_means:.3f}±{std_of_means:.3f}"




def create_full_table(experiment_name,proteomics_name):
    dfs = []
    for tissue_type in ["primary","metastatic"]:
        # factorize to load all tables from runs.. more automated less room for bugs... 
        results_dict = [{"60 protein ensemble \cite{chowdhury2023proteogenomic}":[classical_results[0][proteomics_name + tissue_type.capitalize()],"Omics","ViT"]}, {"1k protein ensemble":[classical_results[1][proteomics_name + tissue_type.capitalize()],"Omics","ViT"]}]
        embeddings = ["ViT","OV_ViT","CTransPath"]

        porpoise60_ensemble = []
        porpoise1k_ensemble = []
        clam_ensemble = []
        
        mcat_60_ensemble = []
        mcat_1k_ensemble = []
        mcat_plat_ensemble = []
        mcat_IPS_pathways = []

        survpath_60_ensemble = []
        survpath_1k_ensemble = []
        survpath_plat_ensemble = []
        survpath_IPS_pathways = []


        for embedding in embeddings: 

            results_dict.append({"clam\_sb \cite{lu2021data}":[experiment_name+"_clam_sb_None_"+embedding+"_"+tissue_type+"_s1","WSI",embedding]})
            
            results_dict.append({"PorpoiseMMF 60 \cite{chen2022pan}":[experiment_name+"_PorpoiseMMF_concat_60_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
            results_dict.append({"PorpoiseMMF 1k \cite{chen2022pan}":[experiment_name+"_PorpoiseMMF_concat_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

            results_dict.append({"MCAT 60 \cite{chen2021multimodal}":[experiment_name+"_MCAT_Surv_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
            results_dict.append({"MCAT 1k \cite{chen2021multimodal}":[experiment_name+"_MCAT_Surv_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
            results_dict.append({"MCAT plat\_resp \cite{chen2021multimodal}":[experiment_name+"_MCAT_Surv_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
            results_dict.append({"MCAT IPS_pathways \cite{chen2021multimodal}":[experiment_name+"_MCAT_Surv_IPS_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})

            results_dict.append({"SurvPath 60 \cite{jaume2023modeling}":[experiment_name+"_SurvPath_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
            results_dict.append({"SurvPath 1k \cite{jaume2023modeling}":[experiment_name+"_SurvPath_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
            results_dict.append({"SurvPath plat\_resp \cite{jaume2023modeling}":[experiment_name+"_SurvPath_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})
            results_dict.append({"SurvPath IPS_pathways \cite{jaume2023modeling}":[experiment_name+"_SurvPath_IPS_pathways_"+embedding+"_"+tissue_type+"_s1","Multimodal",embedding]})


            clam_ensemble.append(experiment_name+"_clam_sb_None_"+embedding+"_"+tissue_type+"_s1")

            porpoise60_ensemble.append(experiment_name+"_PorpoiseMMF_concat_60_"+embedding+"_"+tissue_type+"_s1")
            porpoise1k_ensemble.append(experiment_name+"_PorpoiseMMF_concat_1k_"+embedding+"_"+tissue_type+"_s1")

            mcat_60_ensemble.append(experiment_name+"_MCAT_Surv_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1")
            mcat_1k_ensemble.append(experiment_name+"_MCAT_Surv_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1")
            mcat_plat_ensemble.append(experiment_name+"_MCAT_Surv_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1")
            mcat_IPS_pathways.append(experiment_name+"_MCAT_Surv_IPS_pathways_"+embedding+"_"+tissue_type+"_s1") 

            survpath_60_ensemble.append(experiment_name+"_SurvPath_60_chowdry_clusters_"+embedding+"_"+tissue_type+"_s1")
            survpath_1k_ensemble.append(experiment_name+"_SurvPath_TCGA_grouped_1k_"+embedding+"_"+tissue_type+"_s1")
            survpath_plat_ensemble.append(experiment_name+"_SurvPath_plat_response_pathways_"+embedding+"_"+tissue_type+"_s1")
            survpath_IPS_pathways.append(experiment_name+"_SurvPath_IPS_pathways_"+embedding+"_"+tissue_type+"_s1")

        
        results_dict.append({"clam\_sb \cite{lu2021data}":[clam_ensemble,"WSI","ensemble"]})

        results_dict.append({"PorpoiseMMF 60 \cite{chen2022pan}":[porpoise60_ensemble,"Multimodal","ensemble"]})
        results_dict.append({"PorpoiseMMF 1k \cite{chen2022pan}":[porpoise1k_ensemble,"Multimodal","ensemble"]})

        results_dict.append({"MCAT 60 \cite{chen2021multimodal}":[mcat_60_ensemble,"Multimodal","ensemble"]})
        results_dict.append({"MCAT 1k \cite{chen2021multimodal}":[mcat_1k_ensemble,"Multimodal","ensemble"]})
        results_dict.append({"MCAT plat\_resp \cite{chen2021multimodal}":[mcat_plat_ensemble,"Multimodal","ensemble"]})
        results_dict.append({"MCAT IPS_pathways \cite{chen2021multimodal}":[mcat_IPS_pathways,"Multimodal","ensemble"]})

        results_dict.append({"SurvPath 60 \cite{jaume2023modeling}":[survpath_60_ensemble,"Multimodal","ensemble"]})
        results_dict.append({"SurvPath 1k \cite{jaume2023modeling}":[survpath_1k_ensemble,"Multimodal","ensemble"]})
        results_dict.append({"SurvPath plat\_resp \cite{jaume2023modeling}":[survpath_plat_ensemble,"Multimodal","ensemble"]})
        results_dict.append({"SurvPath IPS_pathways \cite{jaume2023modeling}":[survpath_IPS_pathways,"Multimodal","ensemble"]})


        df = process_auc_files(results_dict)

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
    merged_df_1 = dfs[0].merge(dfs[1], on='model', suffixes=('_primary', '_metastatic'))

    merged_df_1.to_csv("results_tables_dfs/"+experiment_name+"_"+eval+"_"+lr+".csv")

    # Rename the 'name' column after merging
    merged_df_1.drop(['category_primary'], axis=1, inplace=True)
    merged_df_1.rename(columns={'category_metastatic': 'category'}, inplace=True)
    # Get the column you want to move
    column_to_move = merged_df_1.pop('category')
    # Insert the column at the desired position (position 2)
    merged_df_1.insert(1, 'category', column_to_move)

    # Generate LaTeX table
    # caption = "Testing on TCGA samples \cite{cancer2011integrated} AUC scores. All primary tumor samples from the discovery dataset are used for training. Bold values are the highest scores for a given feature extractor and architecture. Underlined are the second-highest scores."
    latex_table = generate_latex_table(merged_df_1,experiment_name,caption="Placeholder")
    # save table to tex file with header name of part/experiment
    with open(save_folder + experiment_name + ".tex", 'w') as f:
        f.write(latex_table)



######## main #########
# need baseline omics scroes.

eval = "test"
lr = "e3"
save_folder = "tables/"+lr+"/"+eval+"/"

with open('classical_results_stratified.json', 'r') as json_file:
    classical_results = json.load(json_file)

experiment_name = "HGSOC_TRAIN_TCGA_15"
proteomics_name = "HGSOC_train_TCGA_test_"
create_full_table(experiment_name,proteomics_name)

experiment_name = "TCGA_TRAIN_HGSOC_15"
proteomics_name = "TCGA_train_HGSOC_test_"
create_full_table(experiment_name,proteomics_name)

experiment_name = "HGSOC_UAB_hold_out_15"
proteomics_name = "HGSOC_UAB_hold_out_"
create_full_table(experiment_name,proteomics_name)

experiment_name = "HGSOC_MAYO_hold_out_15"
proteomics_name = "HGSOC_MAYO_hold_out_"
create_full_table(experiment_name,proteomics_name)



