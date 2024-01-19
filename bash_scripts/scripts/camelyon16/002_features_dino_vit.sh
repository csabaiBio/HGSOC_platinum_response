CUDA_VISIBLE_DEVICES=0,1,2 python CLAM/extract_features_fp.py \
--data_h5_dir /tank/WSI_data/CAMELYON/CAMELYON16/CLAM/slides/ \
--data_slide_dir /tank/WSI_data/CAMELYON/CAMELYON16/images \
--csv_path /tank/WSI_data/CAMELYON/CAMELYON16/CLAM/slides/process_list_autogen.csv \
--slide_ext .tif \
--feat_dir /tank/WSI_data/CAMELYON/CAMELYON16/CLAM/CLAM_ViT_features \
--ssl_model