CUDA_VISIBLE_DEVICES=0,1,2 python create_patches_fp.py --source /tank/WSI_data/Ovarian_WSIs/TCGA-OV --save_dir /local_storage/TCGA_OV_CLAM_data/TCGA_OV_patches_level_0 --patch_size 256 --seg --patch --stitch  --patch_level 0 --preset tcga.csv  # 20X magnification as in paper.









# # file processing the embeddings....
CUDA_VISIBLE_DEVICES=0,1,2 python extract_features_fp.py --data_h5_dir /local_storage/TCGA_OV_CLAM_data/SZ_test_patches_level_0_256 --data_slide_dir /local_storage/ --csv_path /local_storage/TCGA_OV_CLAM_data/SZ_test_patches_level_0_256/process_list_autogen.csv --feat_dir /local_storage/TCGA_OV_CLAM_data/test_features_KimiaNet_level_0 --batch_size 1024 --slide_ext .mrxs 

