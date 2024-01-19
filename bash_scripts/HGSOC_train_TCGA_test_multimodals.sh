# exp_name="TCGA_HGSOC_PorpoiseMMF_ViT_primary"

# CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omics_structure PPI_network_clusters --omic_sizes [27,4,3,5,2,2,4,2,2] &

# ### Metastatic runs

# exp_name="TCGA_HGSOC_PorpoiseMMF_ViT_metastatic"

# CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omics_structure PPI_network_clusters --omic_sizes [27,4,3,5,2,2,4,2,2] &



# # ### CLAM sb

# exp_name="TCGA_HGSOC_PorpoiseMMF_OV_ViT_primary"

# CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omics_structure PPI_network_clusters --omic_sizes [27,4,3,5,2,2,4,2,2] &


# ### Metastatic runs

# exp_name="TCGA_HGSOC_PorpoiseMMF_OV_ViT_metastatic"

# CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omics_structure PPI_network_clusters --omic_sizes [27,4,3,5,2,2,4,2,2] &



### Ctrans path 

exp_name="TCGA_HGSOC_PorpoiseMMF_CTransPath_primary"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/CTransPath --embed_dim 768 --omics_structure concat --omic_sizes [27,4,3,5,2,2,4,2,2]  &



### Metastatic runs

exp_name="TCGA_HGSOC_PorpoiseMMF_CTransPath_metastatic"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/CTransPath --embed_dim 768 --omics_structure concat --omic_sizes [27,4,3,5,2,2,4,2,2] &

