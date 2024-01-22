# first line: 80
@memory.cache
def clean_HGSOC():
    """Take clinical and protemic data tables from Chowrdy et.al and organise into form for training and testing."""
    # From Chowdry.et.al: 
    gloabal_prot = pd.read_csv("/mnt/ncshare/ozkilim/BRCA/data/HGSOC_processed_data/FFPE_discovery_globalprotein_imputed.tsv",delimiter='\t')
    #from TCIA archive: 
    clinical = pd.read_excel("/mnt/ncshare/ozkilim/BRCA/data/HGSOC_processed_data/PTRC-HGSOC_List_clincal_data.xlsx")
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

    final_df = result.rename(columns={'File Name': 'slide_id', 'Patient ID': 'case_id'})
    df_unique = final_df.drop_duplicates(subset="slide_id")

    return df_unique 
