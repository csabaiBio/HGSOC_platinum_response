#!/bin/bash

# # List of models and experetiments to run.
models=("clam_sb" "PorpoiseMMF" "PorpoiseMMF" "MCAT_Surv" "SurvPath" "MCAT_Surv" "SurvPath" "MCAT_Surv" "SurvPath" "MCAT_Surv" "SurvPath")
omics_structures=("None" "concat_60" "concat_1k" "60_chowdry_clusters" "60_chowdry_clusters" "TCGA_grouped_1k" "TCGA_grouped_1k" "plat_response_pathways" "plat_response_pathways" "IPS_pathways" "IPS_pathways")


# Embeddings paths TCGA
embeds_TCGA_ViT=/tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT
embeds_TCGA_Ov_ViT=/tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT
embeds_TCGA_CTransPath=/tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/CTransPath
# Embeddings paths HGSOC
embeds_HGSOC_ViT=/tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT 
embeds_HGSOC_Ov_ViT=/tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT 
embeds_HGSOC_CTransPath=/tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath 


# Part 1. HGSOC train TCGA test

# Get the length of the lists
length=${#models[@]}

# Loop through the lists
for ((i=0; i<length; i++)); do
    # Access elements from both lists using the index
    model=${models[$i]}
    omics_structure=${omics_structures[$i]}


    if [ "$omics_structure" == "60_chowdry_clusters" ]; then
    omic_sizes=[1,33,12,14]
    elif [ "$omics_structure" == "TCGA_grouped_1k" ]; then
        omic_sizes=[161,135,343,79,293,226,302]
    elif [ "$omics_structure" == "concat_60" ]; then
        omic_sizes=[60]
    elif [ "$omics_structure" == "concat_1k" ]; then
        omic_sizes=[1539]
    elif [ "$omics_structure" == "plat_response_pathways" ]; then
        omic_sizes=[67,48,29,31,11,43,6,14,15,2,73,39,32,39,11,47,6,25,51,11,16,46,12,40,2,30,30,7,12,19,23,23,10,5,1,11,5,38,13,32,4,22,9,20,13,13,22,5,1,4,15,3,14,12,6,4,4,1,1,4,1,1,10,4,2,5,5,1,1,1,2,1,3,3,2,1,2,1,1,1]
    elif [ "$omics_structure" == "IPS_pathways" ]; then
        omic_sizes=[14,133,25,14,21,8,50,35,9,1,14,5,92,208,21,82,35,24,138,14,31,105,20,7,6,79,74,80,25,21,90,8,7,50,58,60]
    else
        omic_sizes=[1]
    fi

    # Priamry tumors 
    ##### ViT 
    exp_name="HGSOC_TRAIN_TCGA_15_${model}_${omics_structure}_ViT_primary"
    CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_train_TCGA_test_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_ViT $embeds_HGSOC_ViT  --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes  &
    # ##### Ov_Vit
    exp_name="HGSOC_TRAIN_TCGA_15_${model}_${omics_structure}_OV_ViT_primary"
    CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_train_TCGA_test_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_Ov_ViT $embeds_HGSOC_Ov_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes  &
    #### Ctrans path 
    exp_name="HGSOC_TRAIN_TCGA_15_${model}_${omics_structure}_CTransPath_primary"
    CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_train_TCGA_test_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_CTransPath $embeds_HGSOC_CTransPath --omics_structure $omics_structure --embed_dim 768 --omic_sizes $omic_sizes &

    # Metastatic tumors 
    ##### ViT 
    exp_name="HGSOC_TRAIN_TCGA_15_${model}_${omics_structure}_ViT_metastatic"
    CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_train_TCGA_test_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_ViT $embeds_HGSOC_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes &
    ##### Ov_Vit
    exp_name="HGSOC_TRAIN_TCGA_15_${model}_${omics_structure}_OV_ViT_metastatic"
    CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_train_TCGA_test_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_Ov_ViT $embeds_HGSOC_Ov_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes &
    #### Ctrans path 
    exp_name="HGSOC_TRAIN_TCGA_15_${model}_${omics_structure}_CTransPath_metastatic"
    CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_train_TCGA_test_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_CTransPath $embeds_HGSOC_CTransPath --omics_structure $omics_structure --embed_dim 768 --omic_sizes $omic_sizes 


done

# wait

# # # Part 2. TCGA train HGSOC test

# Loop through the lists
for ((i=0; i<length; i++)); do
    # Access elements from both lists using the index
    model=${models[$i]}
    omics_structure=${omics_structures[$i]}

    if [ "$omics_structure" == "60_chowdry_clusters" ]; then
    omic_sizes=[1,33,12,14]
    elif [ "$omics_structure" == "TCGA_grouped_1k" ]; then
        omic_sizes=[161,135,343,79,293,226,302]
    elif [ "$omics_structure" == "concat_60" ]; then
        omic_sizes=[60]
    elif [ "$omics_structure" == "concat_1k" ]; then
        omic_sizes=[1539]
    elif [ "$omics_structure" == "plat_response_pathways" ]; then
        omic_sizes=[67,48,29,31,11,43,6,14,15,2,73,39,32,39,11,47,6,25,51,11,16,46,12,40,2,30,30,7,12,19,23,23,10,5,1,11,5,38,13,32,4,22,9,20,13,13,22,5,1,4,15,3,14,12,6,4,4,1,1,4,1,1,10,4,2,5,5,1,1,1,2,1,3,3,2,1,2,1,1,1]
    elif [ "$omics_structure" == "IPS_pathways" ]; then
        omic_sizes=[14,133,25,14,21,8,50,35,9,1,14,5,92,208,21,82,35,24,138,14,31,105,20,7,6,79,74,80,25,21,90,8,7,50,58,60]
    else
        omic_sizes=[1]
    fi
    # Priamry tumors 

    ##### ViT 
    exp_name="TCGA_TRAIN_HGSOC_15_${model}_${omics_structure}_ViT_primary"
    CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/TCGA_train_HGSOC_test_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_ViT $embeds_HGSOC_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes  &
    ##### Ov_Vit
    exp_name="TCGA_TRAIN_HGSOC_15_${model}_${omics_structure}_OV_ViT_primary"
    CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/TCGA_train_HGSOC_test_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_Ov_ViT $embeds_HGSOC_Ov_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes &
    #### Ctrans path 
    exp_name="TCGA_TRAIN_HGSOC_15_${model}_${omics_structure}_CTransPath_primary"
    CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/TCGA_train_HGSOC_test_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_CTransPath $embeds_HGSOC_CTransPath --omics_structure $omics_structure --embed_dim 768 --omic_sizes $omic_sizes  &

    # Metastatic tumors 

    ##### ViT 
    exp_name="TCGA_TRAIN_HGSOC_15_${model}_${omics_structure}_ViT_metastatic"
    CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/TCGA_train_HGSOC_test_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_ViT $embeds_HGSOC_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes  &
    ##### Ov_Vit
    exp_name="TCGA_TRAIN_HGSOC_15_${model}_${omics_structure}_OV_ViT_metastatic"
    CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/TCGA_train_HGSOC_test_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_Ov_ViT $embeds_HGSOC_Ov_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes  &
    #### Ctrans path 
    exp_name="TCGA_TRAIN_HGSOC_15_${model}_${omics_structure}_CTransPath_metastatic"
    CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/TCGA_train_HGSOC_test_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_CTransPath $embeds_HGSOC_CTransPath --omics_structure $omics_structure --embed_dim 768 --omic_sizes $omic_sizes 


done

 
wait


# Part 3. HGSOC hospital hold out splits

Loop through the lists
for ((i=0; i<length; i++)); do
    # Access elements from both lists using the index
    model=${models[$i]}
    omics_structure=${omics_structures[$i]}


    if [ "$omics_structure" == "60_chowdry_clusters" ]; then
    omic_sizes=[1,33,12,14]
    elif [ "$omics_structure" == "TCGA_grouped_1k" ]; then
        omic_sizes=[161,135,343,79,293,226,302]
    elif [ "$omics_structure" == "concat_60" ]; then
        omic_sizes=[60]
    elif [ "$omics_structure" == "concat_1k" ]; then
        omic_sizes=[1539]
    elif [ "$omics_structure" == "plat_response_pathways" ]; then
        omic_sizes=[67,48,29,31,11,43,6,14,15,2,73,39,32,39,11,47,6,25,51,11,16,46,12,40,2,30,30,7,12,19,23,23,10,5,1,11,5,38,13,32,4,22,9,20,13,13,22,5,1,4,15,3,14,12,6,4,4,1,1,4,1,1,10,4,2,5,5,1,1,1,2,1,3,3,2,1,2,1,1,1]
    elif [ "$omics_structure" == "IPS_pathways" ]; then
        omic_sizes=[14,133,25,14,21,8,50,35,9,1,14,5,92,208,21,82,35,24,138,14,31,105,20,7,6,79,74,80,25,21,90,8,7,50,58,60]
    else
        omic_sizes=[1]
    fi

    # UAB hold out

    # Priamry tumors 

    ##### ViT 
    exp_name="HGSOC_UAB_hold_out_15_${model}_${omics_structure}_ViT_primary"
    CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_UAB_hold_out_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_ViT $embeds_HGSOC_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes &
    ##### Ov_Vit
    exp_name="HGSOC_UAB_hold_out_15_${model}_${omics_structure}_OV_ViT_primary"
    CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_UAB_hold_out_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_Ov_ViT $embeds_HGSOC_Ov_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes &
    #### Ctrans path 
    exp_name="HGSOC_UAB_hold_out_15_${model}_${omics_structure}_CTransPath_primary"
    CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_UAB_hold_out_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_CTransPath $embeds_HGSOC_CTransPath --omics_structure $omics_structure --embed_dim 768 --omic_sizes $omic_sizes &

    # Metastatic tumors 

    ##### ViT 
    exp_name="HGSOC_UAB_hold_out_15_${model}_${omics_structure}_ViT_metastatic"
    CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_UAB_hold_out_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_ViT $embeds_HGSOC_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes  &
    ##### Ov_Vit
    exp_name="HGSOC_UAB_hold_out_15_${model}_${omics_structure}_OV_ViT_metastatic"
    CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_UAB_hold_out_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_Ov_ViT $embeds_HGSOC_Ov_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes  &
    #### Ctrans path 
    exp_name="HGSOC_UAB_hold_out_15_${model}_${omics_structure}_CTransPath_metastatic"
    CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_UAB_hold_out_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_CTransPath $embeds_HGSOC_CTransPath --omics_structure $omics_structure --embed_dim 768 --omic_sizes $omic_sizes  &


    # Mayo hold out
    
    # Priamry tumors 

    ##### ViT 
    exp_name="HGSOC_MAYO_hold_out_15_${model}_${omics_structure}_ViT_primary"
    CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_MAYO_hold_out_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_ViT $embeds_HGSOC_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes  &
    ##### Ov_Vit
    exp_name="HGSOC_MAYO_hold_out_15_${model}_${omics_structure}_OV_ViT_primary"
    CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_MAYO_hold_out_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_Ov_ViT $embeds_HGSOC_Ov_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes  &
    #### Ctrans path 
    exp_name="HGSOC_MAYO_hold_out_15_${model}_${omics_structure}_CTransPath_primary"
    CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_MAYO_hold_out_Primary --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_CTransPath $embeds_HGSOC_CTransPath --omics_structure $omics_structure --embed_dim 768 --omic_sizes $omic_sizes  &

    Metastatic tumors 

    #### ViT 
    exp_name="HGSOC_MAYO_hold_out_15_${model}_${omics_structure}_ViT_metastatic"
    CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_MAYO_hold_out_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_ViT $embeds_HGSOC_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes &
    ##### Ov_Vit
    exp_name="HGSOC_MAYO_hold_out_15_${model}_${omics_structure}_OV_ViT_metastatic"
    CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_MAYO_hold_out_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_Ov_ViT $embeds_HGSOC_Ov_ViT --embed_dim 384 --omics_structure $omics_structure --omic_sizes $omic_sizes &
    #### Ctrans path 
    exp_name="HGSOC_MAYO_hold_out_15_${model}_${omics_structure}_CTransPath_metastatic"
    CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_platinum_response_prediction --split_dir HGSOC_platinum_responce/splits/HGSOC_MAYO_hold_out_Metastatic --exp_code $exp_name --log_data --model_type $model --max_epoch 15 --weighted_sample --k 5 --embeddings_path $embeds_TCGA_CTransPath $embeds_HGSOC_CTransPath --omics_structure $omics_structure --embed_dim 768 --omic_sizes $omic_sizes 

done

