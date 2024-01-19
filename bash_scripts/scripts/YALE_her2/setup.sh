CUDA_VISIBLE_DEVICES=0 python CLAM/create_patches_fp.py --source /tank/WSI_data/Yale_breast_Her_status/slides  --save_dir /tank/WSI_data/Yale_breast_Her_status/CLAM/level_0/patches --patch_size 224 --preset CLAM/presets/tcga.csv --seg --patch --stitch --patch_level 0 --no_auto_skip

CUDA_VISIBLE_DEVICES=2 python CLAM/extract_features_fp.py  --data_h5_dir /tank/WSI_data/Yale_breast_Her_status/CLAM/level_0/patches  --data_slide_dir /tank/WSI_data/Yale_breast_Her_status/slides --csv_path /tank/WSI_data/Yale_breast_Her_status/CLAM/level_0/patches/process_list_autogen.csv --slide_ext .svs --feat_dir /tank/WSI_data/Yale_breast_Her_status/CLAM/level_0/level_0_ViT --ssl_model 

