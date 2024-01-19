python CLAM/create_patches_fp.py --source /tank/WSI_data/CRC_multiplexed/HandE_slides/processed_RGB/seg_debug/ --save_dir /tank/WSI_data/CRC_multiplexed/CLAM/level_0/slides_debug --patch_size 224 --preset CLAM/presets/tcga.csv --seg --patch --patch_level 0 --no_auto_skip


CUDA_VISIBLE_DEVICES=0 python CLAM/extract_features_fp.py  --data_h5_dir /tank/WSI_data/CRC_multiplexed/CLAM/level_0/slides_debug/ --data_slide_dir /tank/WSI_data/CRC_multiplexed/HandE_slides/processed_RGB/seg_debug/ --csv_path /tank/WSI_data/CRC_multiplexed/CLAM/level_0/slides_debug/process_list_autogen.csv --slide_ext .tif --feat_dir /tank/WSI_data/CRC_multiplexed/CLAM/level_0/custom_ssl_model --ssl_model 


# TODO:  make script for download and clean, extract and embed in one go


# aws s3 ls s3://lin-2021-crc-atlas/data/
# CRC02-HE.ome.tif



