
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import os
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold
from joblib import Memory
from sklearn.model_selection import GroupKFold


cachedir = './cache'  # Define a directory where the cache will be stored
memory = Memory(cachedir, verbose=0)

# This script takes raw published data from the studies used in the paper and creates the full pre-processed dataframe for training all models on.
# This script then creates all splits for all tasks.

def custom_join(df1, df2, column1, column2):
    # Create a new DataFrame to store the result
    df1[column1] = df1[column1].astype(str)
    df2[column2] = df2[column2].astype(str)

    # Create a new DataFrame to store the result
    result_df = pd.DataFrame()

    for index, row in df1.iterrows():
        matching_rows = df2[df2[column2].apply(lambda x: x in row[column1])]
        for _, match in matching_rows.iterrows():
            result_row = pd.concat([row, match])
            result_df = result_df.append(result_row, ignore_index=True)

    return result_df

@memory.cache
def clean_TCGA(TCGA_OV_slides_folder_path):
    """Take clinical and protemic data tables from Zhang et.al and organise into form for training and testing."""
    df_ov_clinical = pd.read_excel("../data/HGSOC_Zhang_TCGA_CPTAC_OV/mmc2.xlsx")
    # select tumors with set of stages
    selected_stages= ["IIIA","IIIB","IIIC","IV"]
    df_ov_clinical = df_ov_clinical[df_ov_clinical["tumor_stage"].isin(selected_stages)]
    # Drop tumors without platinum status
    df_ov_clinical = df_ov_clinical[df_ov_clinical['PlatinumStatus'] != "Not available"]
    # remove not avalable... 
    df_ov_clinical['label'] = df_ov_clinical['PlatinumStatus'].map({'Sensitive': 1, 'Resistant': 0})
    # Load proteomics data:
    tcga_proteomic = pd.read_excel("../data/HGSOC_Zhang_TCGA_CPTAC_OV/mmc3-2.xlsx")
    tcga_prots = tcga_proteomic["hgnc_symbol"].to_list()
    # transopose and organise 
    tcga_proteomic_t = tcga_proteomic.T
    tcga_proteomic_t.columns = tcga_proteomic_t.iloc[1]
    tcga_proteomic_t = tcga_proteomic_t.iloc[2:]
    # impute 
    knn_imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    # Fit and transform the DataFrame
    tcga_proteomic_data = pd.DataFrame(scaler.fit_transform(tcga_proteomic_t), columns=tcga_proteomic_t.columns, index=tcga_proteomic_t.index)
    tcga_proteomic_data = pd.DataFrame(knn_imputer.fit_transform(tcga_proteomic_data), columns=tcga_proteomic_data.columns, index =tcga_proteomic_data.index)
    tcga_proteomic_data["bcr_patient_barcode"] = tcga_proteomic_t.index.str.split('-', 1).str[1]
    # merge cilicnal and protemoic data 
    merged = pd.merge(tcga_proteomic_data,df_ov_clinical, on='bcr_patient_barcode')
    # Merge slides into df
    # List all filenames in the folder
    filenames = [f for f in os.listdir(TCGA_OV_slides_folder_path) if os.path.isfile(os.path.join(TCGA_OV_slides_folder_path, f))]
    # Extract TCGA identifiers from filenames and create a new DataFrame
    # Assuming the TCGA ID is always at the start of the filename up to the third '-'
    file_df = pd.DataFrame({'Filename': filenames})
    file_df['bcr_patient_barcode'] = file_df['Filename'].apply(lambda x: '-'.join(x.split('-')[:3]))
    proteomics_and_wsi = merged.merge(file_df, on='bcr_patient_barcode', how='left')
    proteomics_and_wsi["Filename"] = proteomics_and_wsi["Filename"].str[:-4]
    proteomics_and_wsi = proteomics_and_wsi.rename(columns={"bcr_patient_barcode":"case_id","Filename":"slide_id",'age_at_diagnosis':'Patient Age','tumor_stage':'Tumor Substage','tumor_grade':'Tumor Grade','ethnicity':'Patient Ethnicity'})
    proteomics_and_wsi["Sample Source"]="TCGA"
    proteomics_and_wsi['Tumor type'] = "Primary"    
    proteomics_and_wsi['Tumor Location Group'] = "OV"
    # merge TCGA defined subgroups for patients.
    proteomic_subtypes = pd.read_excel("../data/HGSOC_Zhang_TCGA_CPTAC_OV/1-s2.0-S0092867416306730-mmc5.xlsx", sheet_name='OvarianProteomicClusters')
    proteomic_subtypes = proteomic_subtypes.rename(columns={'Tumor':'case_id'})

    merged = pd.merge(proteomics_and_wsi,proteomic_subtypes, on='case_id',how='left')

    return merged


@memory.cache
def clean_HGSOC():
    """Take clinical and protemic data tables from Chowrdy et.al and organise into form for training and testing."""
    # From Chowdry.et.al: 
    gloabal_prot = pd.read_csv("../data/HGSOC_processed_data/FFPE_discovery_globalprotein_imputed.tsv",delimiter='\t')
    #from TCIA archive: 
    clinical = pd.read_excel("../data/HGSOC_processed_data/PTRC-HGSOC_List_clincal_data.xlsx")
    gloabal_prot_t = gloabal_prot.T
    # drop middle rows .... make header the prot name... 
    gloabal_prot_t.columns = gloabal_prot_t.iloc[1]
    gloabal_prot_t = gloabal_prot_t.iloc[9:]
    row_to_subtract = gloabal_prot_t.iloc[0]
    # Subtract this row from all other rows
    df_subtracted = gloabal_prot_t.subtract(row_to_subtract, axis='columns')
    df_subtracted = df_subtracted.iloc[1:]
    # norm standaard scalar
    scaler = StandardScaler()
    # Fit and transform the DataFrame
    df_subtracted = pd.DataFrame(scaler.fit_transform(df_subtracted), columns=df_subtracted.columns, index=df_subtracted.index)
    df_subtracted["Sample ID1"] = df_subtracted.index
    ### merge on column of TCGA Proteomic subtype catagories.
    tcga_clusters = pd.read_excel('../data/HGSOC_processed_data/mmc4.xlsx',sheet_name="Inferred_TCGA_clusters_FD_prot")
    tcga_clusters = tcga_clusters.rename(columns={'Subtype':'Proteomic subtype','Sample':'Sample ID1'})
    tcga_clusters['Proteomic subtype'] = tcga_clusters['Proteomic subtype'].map({'IMR':'Immunoreactive','DIF':'Differentiated','MES':'Mesenchymal','PRO':'Proliferative'}) # align with TCGA names.

    df_subtracted = pd.merge(df_subtracted,tcga_clusters, on='Sample ID1',how='left')
    result = custom_join(df_subtracted, clinical, 'Sample ID1', 'Sample ID')
    result['label'] = result['Tumor response'].map({'sensitive': 1, 'refractory': 0, 'Sensitive': 1, 'Refractory': 0})

    final_df = result.copy()

    final_df = result.rename(columns={'File Name':'slide_id', 'Patient ID':'case_id'})
    df_unique = final_df.drop_duplicates(subset="slide_id")

    return df_unique 



def save_splits(main_df,task_type,tumor_location):
    """Given a task type create spl;its to be used for experiments."""
    # Focus on the 'Sample source' and 'slide_id' columns
    data = main_df[['Sample Source', 'slide_id','Tumor type','case_id']]
    # Get unique sample sources
    # sample_sources = data['Sample Source'].unique()

    if task_type =="TCGA_train_HGSOC_test":
        cv_data = data[data['Sample Source']=="TCGA"]['slide_id']
        groups = data[data['Sample Source']=="TCGA"]['case_id'].astype(str)
        data = data[data["Tumor type"]==tumor_location]
        test_data = data[data['Sample Source']!="TCGA"]['slide_id']
        unique_groups = groups.nunique()
        print(f"Number of unique groups: {unique_groups}")

    elif task_type =="HGSOC_train_TCGA_test":
        test_data = data[data['Sample Source']=="TCGA"]['slide_id'] 
        data = data[data["Tumor type"]==tumor_location]
        cv_data = data[data['Sample Source']!="TCGA"]['slide_id']
        groups = data[data['Sample Source']!="TCGA"]['case_id'].astype(str)

        unique_groups = groups.nunique()
        print(f"Number of unique groups: {unique_groups}")

    elif task_type =="HGSOC_MAYO_hold_out":
        data = data[data["Tumor type"]==tumor_location]
        cv_data = data[data['Sample Source'].isin(["UAB","FHCRC"])]['slide_id']
        test_data = data[data['Sample Source']=="Mayo"]['slide_id']
        groups = data[data['Sample Source'].isin(["UAB","FHCRC"])]['case_id'].astype(str)

        unique_groups = groups.nunique()
        print(f"Number of unique groups: {unique_groups}")  

    elif task_type =="HGSOC_UAB_hold_out":
        data = data[data["Tumor type"]==tumor_location]
        cv_data = data[data['Sample Source'].isin(["Mayo","FHCRC"])]['slide_id']
        test_data = data[data['Sample Source']=="UAB"]['slide_id']
        groups = data[data['Sample Source'].isin(["Mayo","FHCRC"])]['case_id'].astype(str)
        unique_groups = groups.nunique()
        print(f"Number of unique groups: {unique_groups}")


    # Create a KFold object for 5 folds
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Directory to store split files

    kf = GroupKFold(n_splits=5)
    # The groups parameter should be the 'case_id' column of your DataFrame
    # Apply 5-fold cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(cv_data, groups=groups)):
        # Extract train, val, and test slide IDs
        train_ids = cv_data.iloc[train_index].tolist()
        val_ids = cv_data.iloc[val_index].tolist()
        test_ids = test_data.tolist()

        # Determine the maximum length among train, val, and test lists
        max_len = max(len(train_ids), len(val_ids), len(test_ids))

        # Extend lists to have the same length
        train_ids.extend([""] * (max_len - len(train_ids)))
        val_ids.extend([""] * (max_len - len(val_ids)))
        test_ids.extend([""] * (max_len - len(test_ids)))

        # Create the split DataFrame
        split_df = pd.DataFrame({'train': train_ids[:max_len], 'val': val_ids[:max_len], 'test': test_ids[:max_len]})
        # Save the split file
        split_folder = f'./splits/{task_type}_{tumor_location}'
        os.makedirs(split_folder, exist_ok=True)
        split_filename = f'{split_folder}/splits_{fold}.csv'
        # split_filepath = os.path.join(split_dir, split_filename)
        split_df.to_csv(split_filename, index=True)

    print("Split files created successfully.")



def check_duplicate_columns(df):
    duplicate_columns = df.columns[df.columns.duplicated()]
    return duplicate_columns
    
def remove_duplicate_columns(df):
    return df.loc[:, ~df.columns.duplicated()]

def join_all_data(TCGA_table,HGSOC_table):
    """Join both data sources into one main dataframe where all experiments can be run from"""
    # Join based on overlapping proteins and metadata column names.

    TCGA_table = remove_duplicate_columns(TCGA_table)
    common_columns = sorted(HGSOC_table.columns.intersection(TCGA_table.columns))

    print(TCGA_table["Sample Source"])
    print(HGSOC_table["Sample Source"])

    # Select only common columns from both dataframes
    HGSOC_intersection = HGSOC_table[common_columns]
    TCGA_intersection = TCGA_table[common_columns]
    # Use the function on your DataFrame
    duplicate_columns = check_duplicate_columns(TCGA_table)

    if len(duplicate_columns) > 0:
        print("Duplicate columns found:", duplicate_columns.tolist())
    else:
        print("No duplicate columns found.")

    assert HGSOC_intersection.columns.to_list() == TCGA_intersection.columns.to_list()
    combined_df = pd.concat([HGSOC_intersection, TCGA_intersection], ignore_index=True)

    print(combined_df["Sample Source"].value_counts())
    print(combined_df["Patient Age"].value_counts())
    print(combined_df["Proteomic subtype"].value_counts())
    print(combined_df['Tumor type'].value_counts())
    
    return combined_df



if __name__ == "__main__":
    ###### Join all datasets ######
    print("Cleaning TCGA data")
    TCGA_OV_slides_folder_path = '/tank/WSI_data/Ovarian_WSIs/TCGA-OV/slides' #path to where you have saved the TCGA .svs WSIs
    TCGA_table = clean_TCGA(TCGA_OV_slides_folder_path)
    print("Cleaning HGSOC data")
    HGSOC_table = clean_HGSOC()
    print("Merging all data")
    main_df = join_all_data(TCGA_table,HGSOC_table)
    main_df.to_csv("HGSOC_TCGA_main.csv",index=None)
    print("creating splits")
    ###### Create splits ######
    for tumor_location in ["Primary","Metastatic"]:
        # Split: 1: TCGA test HGSOC train
        save_splits(main_df,"TCGA_train_HGSOC_test",tumor_location)
        # Split: 2: HGSOC train TCGA test
        save_splits(main_df,"HGSOC_train_TCGA_test",tumor_location)
        # Split: 3: HGSOC MAYO hold out
        save_splits(main_df,"HGSOC_MAYO_hold_out",tumor_location)
        # Split: 4: HGSOC UBC hold out
        save_splits(main_df,"HGSOC_UAB_hold_out",tumor_location)

    print("done.")
