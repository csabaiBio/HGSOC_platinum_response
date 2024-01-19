# Script that patches all ovarian slides for training of a bulk Ovarain SSL pre trained network. 


# TCGA deffo correct patched?... 

# TCGA (running)
python CLAM/WSI_to_patch_bags.py --csv_path /mnt/ncshare/ozkilim/BRCA/data/bulk_summary_lists/TCGA_all.csv --data_slide_dir /tank/WSI_data/Ovarian_WSIs/TCGA-OV  --data_h5_dir /local_storage/Ovarain_CLAM_data/TCGA_OV_patches_level_0 --save_path bulk_patches --slide_ext .svs
# HGSC (running)
python CLAM/WSI_to_patch_bags.py --csv_path /mnt/ncshare/ozkilim/BRCA/data/bulk_summary_lists/HGSOC_all.csv --data_slide_dir /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian  --data_h5_dir /local_storage/Ovarain_CLAM_data/HGSOC/patches_level_0 --save_path bulk_patches --slide_ext .svs


# CUDA_VISIBLE_DEVICES=0,1,2 python create_patches_fp.py --source /tank/WSI_data/Ovarian_WSIs/CPTAC_OV --save_dir /local_storage/Ovarain_CLAM_data/CPTAC-OV/patches_level_0 --patch_size 256 --seg --patch --stitch  --patch_level 0 --preset presets/tcga.csv  # 20X magnification as in paper.

# CPTAC_OV then make images....
python CLAM/WSI_to_patch_bags.py --csv_path /mnt/ncshare/ozkilim/BRCA/data/bulk_summary_lists/CPTAV-OV_all.csv --data_slide_dir /tank/WSI_data/Ovarian_WSIs/CPTAC_OV  --data_h5_dir /local_storage/Ovarain_CLAM_data/CPTAC-OV/patches_level_0 --save_path bulk_patches --slide_ext .svs
# Bevacizumab

# CUDA_VISIBLE_DEVICES=0,1,2 python create_patches_fp.py --source /tank/WSI_data/Ovarian_WSIs/Ovarian_combined_tif --save_dir /local_storage/Ovarain_CLAM_data/Bevacizumab/patches_level_0 --patch_size 256 --seg --patch --stitch  --patch_level 0 --preset tcga.csv  # 20X magnification as in paper.

python CLAM/WSI_to_patch_bags.py --csv_path /mnt/ncshare/ozkilim/BRCA/data/bulk_summary_lists/Bev_all.csv --data_slide_dir /tank/WSI_data/Ovarian_WSIs/Ovarian_combined_tif  --data_h5_dir /local_storage/Ovarain_CLAM_data/Bevacizumab/patches_level_0 --save_path bulk_patches --slide_ext .tif


# UBC
# # possible with png?....
# CUDA_VISIBLE_DEVICES=0,1,2 python create_patches_fp.py --source /home/qbeer/datasets/UBC-OCEAN --save_dir /local_storage/Ovarain_CLAM_data/UBC/patches_level_0 --patch_size 256 --seg --patch --stitch  --patch_level 0 --preset presets/tcga.csv  # 20X magnification as in paper.



python3 CLAM/WSI_to_patch_bags.py --csv_path /mnt/ncshare/ozkilim/BRCA/data/bulk_summary_lists/UBC_all.csv --data_slide_dir /home/qbeer/datasets/UBC-OCEAN  --data_h5_dir /local_storage/Ovarain_CLAM_data/UBC/patches_level_0 --save_path bulk_patches --slide_ext .tif



