#!/bin/bash

# Define a list of variables
models=("MCAT_Surv" "SurvPath")
omics_structures=("60_chowdry_clusters" "60_chowdry_clusters")


# Get the length of the lists
length=${#models[@]}

# Loop through the lists
for ((i=0; i<length; i++)); do
    # Access elements from both lists using the index
    model=${models[$i]}
    omics_structure=${omics_structures[$i]}
    ##### ViT 
    exp_name="TCGA_TRAIN_HGSOC_50_${model}_ViT_primary"
    CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_train_HGSOC_primary_Test --exp_code $exp_name --log_data --model_type $model --max_epoch 50 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes [1,33,12,14] &
    ##### Ov_Vit
    exp_name="TCGA_TRAIN_HGSOCC_50_${model}_OV_ViT_primary"
    CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_train_HGSOC_primary_Test --exp_code $exp_name --log_data --model_type $model --max_epoch 50 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes [1,33,12,14] &
    #### Ctrans path 
    exp_name="TCGA_TRAIN_HGSOCC_50_${model}_CTransPath_primary"
    CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_train_TCGA_test --split_dir splits/TCGA_train_HGSOC_primary_Test --exp_code $exp_name --log_data --model_type $model --max_epoch 50 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/CTransPath --omics_structure $omics_structure --embed_dim 768 --omic_sizes [1,33,12,14] &
done

