CUDA_VISIBLE_DEVICES=0 python CLAM/create_patches_fp.py --source /tank/WSI_data/bracs_icar/slides  --save_dir /tank/WSI_data/bracs_icar/CLAM/level_0/patches --patch_size 224 --preset CLAM/presets/tcga.csv --seg --patch --stitch --patch_level 0 --no_auto_skip


#SETUP embedding of bracs!

CUDA_VISIBLE_DEVICES=2 python CLAM/extract_features_fp.py  --data_h5_dir /tank/WSI_data/bracs_icar/CLAM/level_0/patches  --data_slide_dir /tank/WSI_data/bracs_icar/slides  --csv_path /tank/WSI_data/bracs_icar/CLAM/level_0/patches/process_list_autogen.csv --slide_ext .svs --feat_dir /tank/WSI_data/bracs_icar/CLAM/level_0/level_0_ViT --ssl_model 

