# exp_name="TCGA_HGSOC_clam_sb_ViT_primary"

# CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_primary --exp_code $exp_name --log_data --model_type clam_sb --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omic_sizes [1,2,33,14,14,89] &

# ### Metastatic runs

# exp_name="TCGA_HGSOC_clam_sb_ViT_metastatic"

# CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_metastatic --exp_code $exp_name --log_data --model_type clam_sb --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omic_sizes [1,2,33,14,14,89] &



# # ### CLAM sb

# exp_name="TCGA_HGSOC_clam_sb_OV_ViT_primary"

# CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_primary --exp_code $exp_name --log_data --model_type clam_sb --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omic_sizes [1,2,33,14,14,89] &


# ### Metastatic runs

# exp_name="TCGA_HGSOC_clam_sb_OV_ViT_metastatic"

# CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_metastatic --exp_code $exp_name --log_data --model_type clam_sb --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omic_sizes [1,2,33,14,14,89] &



### Ctrans path 

exp_name="TCGA_HGSOC_clam_sb_CTransPath_primary"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_primary --exp_code $exp_name --log_data --model_type clam_sb --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [1,2,33,14,14,89]  &



### Metastatic runs

exp_name="TCGA_HGSOC_clam_sb_CTransPath_metastatic"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_metastatic --exp_code $exp_name --log_data --model_type clam_sb --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [1,2,33,14,14,89] &

