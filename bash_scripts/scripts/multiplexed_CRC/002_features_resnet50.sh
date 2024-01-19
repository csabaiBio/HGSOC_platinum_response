CUDA_VISIBLE_DEVICES=0,1,2 python CLAM/extract_features_fp.py \
--data_h5_dir /tank/nc_share/qbeer/scientificdata_to_be_submitted_revision/CLAM/CLAM_slides/ \
--data_slide_dir /tank/nc_share/qbeer/scientificdata_to_be_submitted_revision/slides/ \
--csv_path /tank/nc_share/qbeer/scientificdata_to_be_submitted_revision/CLAM/CLAM_slides/process_list_autogen.csv \
--slide_ext .mrxs \
--feat_dir /tank/nc_share/qbeer/scientificdata_to_be_submitted_revision/CLAM/CLAM_ResNet50_features