# creation of main manuscript tables.
import matplotlib.pylab as plt
import glob
import os
import numpy as np
import pandas as pd 
import re

# code to filter down the tables and find best overall.
    
def extract_mean(value):
    """
    Extract the mean value from a performance metric string.
    """
    mean, _ = value.split("±")
    return float(mean)

def flatten_and_filter_multimodal(df,type):
    """
    Flatten a table to have rows for each architecture-backbone combination,
    specifically filtering for 'Multimodal' category models.
    """
    # Filter for 'Multimodal' models before flattening
    df = df[["category_"+type,"model","TCGA CTransPath_"+type,"TCGA OV_ViT_"+type,"TCGA ViT_"+type]]
    filtered_df = df[df['category_'+type] == 'Multimodal']    
    melted = filtered_df.melt(id_vars=['model', 'category_'+type], var_name='backbone', value_name='performance')
    melted['mean_performance'] = melted['performance'].apply(extract_mean)
    melted.drop(columns=['performance'], inplace=True)
    return melted

def aggregate_performance(flattened_tables):
    """
    Aggregate performance across multiple tables and rank architecture-backbone combinations.
    """
    combined = pd.concat(flattened_tables)
    combined['rank_metric'] = combined.groupby(['model', 'backbone'])['mean_performance'].transform('mean')
    ranked = combined.drop_duplicates(['model', 'backbone']).sort_values(by='rank_metric', ascending=False).reset_index(drop=True)
    ranked = ranked.drop(columns="mean_performance")
    return ranked

def get_max_lr_results(experiment_name,eval):
    dataframes = []
    for lr in ["5e3","5e4","10e4"]:
        df = pd.read_csv("/mnt/ncshare/ozkilim/BRCA/HGSOC_platinum_responce/results_tables_dfs/"+ experiment_name + "_" + eval + "_" + lr + ".csv" )
        df = df.drop(columns=["Unnamed: 0"])
        dataframes.append(df)
        # create "best" df by picking the highest values for each model across the 3 dataframes. (Save the lerning rate aka lr in another column...)
        combined_df = pd.concat(dataframes, axis=0)
        # Group by the index (assuming row labels are identical and meaningful across DataFrames)
        # and take the max of each group to find the largest value for each cell
    max_df = combined_df.groupby(combined_df.index).max()

    return max_df 

def df_to_latex_table(df):
    # Define the header of the LaTeX table with the correct experiment names and structure
    # Define the header of the LaTeX table with the correct experiment names and structure
    experiments = ['HGSOC_TRAIN_TCGA_15', 'TCGA_TRAIN_HGSOC_15', 'HGSOC_UAB_hold_out_15']
    test_types = ['test', 'val']
    
    header = r"""\begin{table}[ht]
    \footnotesize
    \centering
    \begin{tabular}{l*{6}{S[table-format=1.8, table-align-text-post=false]}}
    \toprule
    \multicolumn{1}{c}{Model} & \multicolumn{2}{c}{HGSOC Train} & \multicolumn{2}{c}{TCGA Train} & \multicolumn{2}{c}{HGSOC Train} \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
    & \multicolumn{1}{c}{TCGA Test} & \multicolumn{1}{c}{5 fold CV} & \multicolumn{1}{c}{HGSOC Test} & \multicolumn{1}{c}{5 fold CV} & \multicolumn{1}{c}{UAB Test} & \multicolumn{1}{c}{5 fold CV} \\
    \midrule
    """
    footer = r"""\bottomrule
    \end{tabular}
    \vspace{6pt}
    \caption{Main results table.}
    \label{tab:main}
    \end{table}"""

    # Preprocess DataFrame to identify and format the largest and second largest values
    formatted_results = {}
    for exp in experiments:
        for test_type in test_types:
            col = df[(exp, test_type)]
            # Remove '±' and convert to float for comparison, handling missing values
            numeric_values = col.str.split('±').str[0].astype(float)
            largest = numeric_values.nlargest(2)
            
            formatted_results[(exp, test_type, largest.index[0])] = r"\textbf{" + col.iloc[largest.index[0]] + "}"
            if len(largest) > 1:
                formatted_results[(exp, test_type, largest.index[1])] = r"\underline{" + col.iloc[largest.index[1]] + "}"
    
    body = ""
    for index, row in df.iterrows():
        model_name = row['model'].item()
        results = []
        for exp in experiments:
            for test_type in test_types:
                key = (exp, test_type, index)
                if key in formatted_results:
                    result = formatted_results[key]
                else:
                    result = row[(exp, test_type)] if (exp, test_type) in row else 'N/A'
                results.append(result)
        body += f"{model_name} & {' & '.join(results)} \\\\ \n"
    
    return header + body + footer


########## Main #########

experiments = ["HGSOC_TRAIN_TCGA_15","TCGA_TRAIN_HGSOC_15","HGSOC_UAB_hold_out_15"]
eval = "test"
type = "metastatic"

cross_experiment_results = []
for experiment_name in experiments:
    dataframes = []  
    max_df = get_max_lr_results(experiment_name,eval)
    max_df = flatten_and_filter_multimodal(max_df,type)
    cross_experiment_results.append(max_df)
    # print(experiment_name)
    # print(max_df)

ranked = aggregate_performance(cross_experiment_results)

# now wwe have everything to creat the primary table with the best model.
print(ranked)
best_model = ranked[["model","backbone"]].iloc[0]
print(best_model)

main_table = []
experiments = ["HGSOC_TRAIN_TCGA_15","TCGA_TRAIN_HGSOC_15","HGSOC_UAB_hold_out_15"]
for experiment_name in experiments:
    eval = "test"
    max_df = get_max_lr_results(experiment_name,eval)
    result_multimodal = max_df[(max_df["model"] == best_model["model"])][best_model["backbone"]].item()   
    result_WSI = max_df[(max_df["model"] == "clam\_sb \cite{lu2021data}")][best_model["backbone"]].item()   
    result_Omics = max_df[(max_df["model"] == "60 protein ensemble \cite{chowdhury2023proteogenomic}")][best_model["backbone"]].item()   

    main_table.append({"experiment":experiment_name,"test_type":eval,"result":result_multimodal,"model":"Multimodal"})
    main_table.append({"experiment":experiment_name,"test_type":eval,"result":result_WSI,"model":"Histopathology"})
    main_table.append({"experiment":experiment_name,"test_type":eval,"result":result_Omics,"model":"Proteomics"})

    eval = "val"
    max_df = get_max_lr_results(experiment_name,eval)
    result_multimodal = max_df[(max_df["model"] == best_model["model"])][best_model["backbone"]].item()   
    result_WSI = max_df[(max_df["model"] == "clam\_sb \cite{lu2021data}")][best_model["backbone"]].item()   
    result_Omics = max_df[(max_df["model"] == "60 protein ensemble \cite{chowdhury2023proteogenomic}")][best_model["backbone"]].item()   

    main_table.append({"experiment":experiment_name,"test_type":eval,"result":result_multimodal,"model":"Multimodal"})
    main_table.append({"experiment":experiment_name,"test_type":eval,"result":result_WSI,"model":"Histopathology"})
    main_table.append({"experiment":experiment_name,"test_type":eval,"result":result_Omics,"model":"Proteomics"})


# TODO: omics val result need to add! 

df = pd.DataFrame(main_table)
# Pivot the DataFrame
df_pivoted = df.pivot_table(index='model', columns=['experiment', 'test_type'], values='result', aggfunc='first')
# Reorder the columns to ensure they are in the desired paired format
df_pivoted = df_pivoted.reindex(columns=pd.MultiIndex.from_product([
    df['experiment'].unique(), 
    ['test', 'val']
], names=['experiment', 'test_type']))

df_pivoted.reset_index(inplace=True)
df_pivoted.columns.name = None

# Assuming df_pivoted is your pivoted DataFrame
latex_code = df_to_latex_table(df_pivoted)

save_folder = "/mnt/ncshare/ozkilim/BRCA/HGSOC_platinum_responce/tables/main_tables/"+type + ".tex"
with open(save_folder, 'w') as f:
    f.write(latex_code)