
CUDA_VISIBLE_DEVICES=2 python CLAM/create_patches_fp.py --source /tank/WSI_data/Ovarian_WSIs/TCGA-OV  --save_dir /local_storage/Ovarain_CLAM_data/TCGA_OV_ViT/Vit_patches --patch_size 224 --preset CLAM/presets/tcga.csv --seg --patch --patch_level 0

#Run 72 slides of interest through checkpoint! 
CUDA_VISIBLE_DEVICES=2 python CLAM/extract_features_fp.py  --data_h5_dir /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/patches --data_slide_dir /tank/WSI_data/Ovarian_WSIs/TCGA-OV/slides --csv_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/patches/BRCA_pos_neg_FFPE.csv --slide_ext .svs --feat_dir /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --OV_ViT 




#Zoltan test set

CUDA_VISIBLE_DEVICES=1 python CLAM/create_patches_fp.py --source /tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/slides  --save_dir /tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/CLAM/level_2/patches --patch_size 224 --preset CLAM/presets/tcga.csv --seg --patch --stitch --patch_level 2 





CUDA_VISIBLE_DEVICES=2 python CLAM/extract_features_fp.py  --data_h5_dir /tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/CLAM/level_2/patches --data_slide_dir /tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/slides/ --csv_path /tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/CLAM/level_2/patches/process_list_autogen.csv --slide_ext .mrxs --feat_dir /tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/CLAM/level_2/Ov_ViT --OV_ViT 



#FACTORIZE THIS TO MAKE NO MISTAKES!

# HGSOC

CUDA_VISIBLE_DEVICES=2 python CLAM/create_patches_fp.py --source /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/slides  --save_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/patches --patch_size 224 --preset CLAM/presets/tcga.csv --seg --patch --patch_level 0



#Run 72 slides of interest through checkpoint! 
CUDA_VISIBLE_DEVICES=2 python CLAM/extract_features_fp.py  --data_h5_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/patches --data_slide_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/slides --csv_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/patches/process_list_autogen.csv  --slide_ext .svs --feat_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT --OV_ViT 

