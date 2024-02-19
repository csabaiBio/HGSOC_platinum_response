# creation of main manuscript tables.

# code to filter down the tables and find best overall.
    
def extract_mean(value):
    """
    Extract the mean value from a performance metric string.
    """
    mean, _ = value.split("±")
    return float(mean)

def flatten_and_filter_multimodal(df):
    """
    Flatten a table to have rows for each architecture-backbone combination,
    specifically filtering for 'Multimodal' category models.
    """
    # Filter for 'Multimodal' models before flattening
    filtered_df = df[df['category'] == 'Multimodal']
    melted = filtered_df.melt(id_vars=['model', 'category'], var_name='backbone', value_name='performance')
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
    return ranked


# Assuming you have loaded your tables into DataFrames: df1, df2, ...
# Flatten each table and filter for 'Multimodal' models
# flattened_tables = [flatten_and_filter_multimodal(df) for df in [merged_df_1,merged_df_2]]
# # Aggregate performance and get the ranked list
# ranked_combinations = aggregate_performance(flattened_tables)
# print(ranked_combinations)

# from the best create a final figures table to is the summary... 


# Load all tables from first script,

# collate 

# pick best models ranked and gen main tables. 

# MCAT surv for metastatic deseases 

# Porpoise for Primary tumors. 

# ensemble all? ... 



# def gen_main_latex_table():
#     # TODO: take all results table and apply logic to create final aggregate optimal table for main manuscript. 
#     \begin{table}[ht]
#     \footnotesize
#     \centering
#     \begin{tabular}{l*{6}{S[table-format=1.8, table-align-text-post=false]}}
#     \toprule
#     \multicolumn{1}{c}{Primary tumors} & \multicolumn{2}{c}{TCGA train } & \multicolumn{2}{c}{HGSOC train \cite{jaume2023modeling}} & \multicolumn{2}{c}{HGSOC train} \\
#     \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
#     Model & \multicolumn{1}{c}{HGSOC Test} & \multicolumn{1}{c}{5 fold CV} & \multicolumn{1}{c}{TCGA Test} & \multicolumn{1}{c}{5 fold CV } & \multicolumn{1}{c}{Mayo Test} & \multicolumn{1}{c}{5 fold CV} \\
#     \midrule
#     Multimodal \cite{jaume2023modeling}  & 0.638 $\pm$ 0.12 & 0.519 $\pm$ 0.086 & 0.644 $\pm$ 0.113 & 0.531 $\pm$ 0.114 & 0.644 $\pm$ 0.113 & 0.531 $\pm$ 0.114 \\
#     Proteomics \cite{chowdhury2023proteogenomic}  & 0.553 $\pm$ 0.082 & 0.553 $\pm$ 0.082 & 0.553 $\pm$ 0.082 & 0.553 $\pm$ 0.085 & 0.644 $\pm$ 0.113 & 0.531 $\pm$ 0.114 \\
#     Histopathology \cite{lu2021data} & 0.471 $\pm$ 0.082 & \textbf{0.727 $\pm$ 0.038} & 0.601 $\pm$ 0.049 & 0.35 $\pm$ 0.028 & 0.644 $\pm$ 0.113 & 0.531 $\pm$ 0.114 \\
#     \bottomrule
#     \end{tabular}
#     \vspace{6pt}
#     \caption{Main results table.}
#     \label{tab:main}
#     \end{table}
