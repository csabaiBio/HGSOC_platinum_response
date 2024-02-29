# first line: 33
@memory.cache
def clean_TCGA(TCGA_OV_slides_folder_path):
    """Take clinical and protemic data tables from Zhang et.al and organise into form for training and testing."""
    df_ov_clinical = pd.read_excel("../data/HGSOC_Zhang_TCGA_CPTAC_OV/mmc2.xlsx")
    # select tumors with set of stages

    # selected_stages= ["IIIA","IIIB","IIIC","IV"]
    # df_ov_clinical = df_ov_clinical[df_ov_clinical["tumor_stage"].isin(selected_stages)]
    # Drop tumors without platinum status
    df_ov_clinical = df_ov_clinical[df_ov_clinical['PlatinumStatus'] != "Not available"]
    # remove not avalable... 
    df_ov_clinical['label'] = df_ov_clinical['PlatinumStatus'].map({'Sensitive': 1, 'Resistant': 0})
    # Load proteomics data:
    tcga_proteomic = pd.read_excel("../data/HGSOC_Zhang_TCGA_CPTAC_OV/mmc3-2.xlsx")
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
    file_df['slide_type'] = file_df['Filename'].str.extract(r'-([^-\.]+)\.')
    slide_types = ['TS1','DX1','TSA','DX2']
    file_df = file_df[file_df['slide_type'].isin(slide_types)]
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
    df_unique = merged.drop_duplicates(subset="slide_id")

    return df_unique
