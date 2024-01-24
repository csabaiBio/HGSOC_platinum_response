raw_slides_dir="/tank/WSI_data/Ovarian_WSIs/TCGA-OV/slides/"
patch_save_dir="/tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/patches"
csv_path="/tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/patches/proc_list.csv"
slide_ext=".svs"
feat_dir="/tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/CTransPath/"
# target_image="/tank/WSI_data/Ovarian_WSIs/TCGA-OV/slides/TCGA-23-1027-01Z-00-DX1.53F9DFF4-6811-4184-B2FD-1F6706B948FD.svs"

# raw_slides_dir="/tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/slides/"
# patch_save_dir="/tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/CLAM/level_0/patches_stict"
# csv_path="/tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/CLAM/level_0/patches_stict/process_list_autogen.csv"
# slide_ext=".mrxs"
# feat_dir="/tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/CLAM/level_0/Ov_ViT_stain_norm_Mancheko/"
# target_image="/tank/WSI_data/Ovarian_WSIs/TCGA-OV/slides/TCGA-23-1027-01Z-00-DX1.53F9DFF4-6811-4184-B2FD-1F6706B948FD.svs"


# raw_slides_dir="/tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/slides/"
# patch_save_dir="/tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/patches"
# csv_path="/tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/patches/process_list_autogen.csv"
# slide_ext=".svs"
# feat_dir="/tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath/"

# target_image="/tank/WSI_data/Ovarian_WSIs/TCGA-OV/slides/TCGA-23-1027-01Z-00-DX1.53F9DFF4-6811-4184-B2FD-1F6706B948FD.svs"


# raw_slides_dir="/tank/WSI_data/spatial_transcriptomics/slides/"
# patch_save_dir="/tank/WSI_data/spatial_transcriptomics/CLAM/level_0/patches/"
# csv_path="/tank/WSI_data/spatial_transcriptomics/CLAM/level_0/patches/process_list_autogen.csv"
# slide_ext=".tiff"
# feat_dir="/tank/WSI_data/spatial_transcriptomics/CLAM/level_0/ViT/"
# # target_image="/tank/WSI_data/Ovarian_WSIs/TCGA-OV/slides/TCGA-23-1027-01Z-00-DX1.53F9DFF4-6811-4184-B2FD-1F6706B948FD.svs"



# Patching
# CUDA_VISIBLE_DEVICES=0 python CLAM/create_patches_fp.py --source $raw_slides_dir --save_dir $patch_save_dir --patch_size 224 --preset CLAM/presets/bwh_biopsy.csv --seg --patch --stitch --patch_level 0 
# Embedding
CUDA_VISIBLE_DEVICES=0 python CLAM/extract_features_fp.py  --data_h5_dir $patch_save_dir  --data_slide_dir $raw_slides_dir --csv_path $csv_path --slide_ext $slide_ext --feat_dir $feat_dir --CTransPath 