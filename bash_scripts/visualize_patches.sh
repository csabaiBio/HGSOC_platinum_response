# This script produces a few visual samples of for WSI images.

# For Zs test set
python CLAM/get_visual_samples.py --csv_path data/raw_subsets/ZS_test_list_prepared.csv --data_slide_dir /tank/WSI_data/Ovarian_WSIs/BRCA_blind_test  --data_h5_dir data/embeds/SZ_test_patches_level_1_256  --save_folder data/example_patches_test_level1/  --slide_ext .mrxs

# For TCGA train set. 
python CLAM/get_visual_samples.py --csv_path data/raw_subsets/brca_dataset_FFPE_NERO.csv --data_slide_dir /tank/WSI_data/Ovarian_WSIs/TCGA-OV  --data_h5_dir  data/embeds/TCGA_OV_patches_level_0  --save_folder data/example_patches_TCGA/  --slide_ext .svs
