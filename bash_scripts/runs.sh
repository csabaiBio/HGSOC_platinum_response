exp_name="HGSOC_PorpoiseMMF_ViT_Mayo_primary_15_epocs"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/Mayo_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_ViT_UAB_primary_15_epocs"

CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/UAB_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_ViT_FHCRC_primary_15_epocs"

CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/FHCRC_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT --omic_sizes [1,2,33,14,14,89] &


### Metastatic runs

exp_name="HGSOC_PorpoiseMMF_ViT_Mayo_metastatic_15_epocs"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/Mayo_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_ViT_UAB_metastatic_15_epocs"

CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/UAB_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_ViT_FHCRC_metastatic_15_epocs"

CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/FHCRC_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/ViT --omic_sizes [1,2,33,14,14,89] &



# ### CLAM sb

exp_name="HGSOC_PorpoiseMMF_OV_ViT_Mayo_primary_15_epocs"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/Mayo_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_OV_ViT_UAB_primary_15_epocs"

CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/UAB_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_OV_ViT_FHCRC_primary_15_epocs"

CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/FHCRC_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT --omic_sizes [1,2,33,14,14,89] &


### Metastatic runs

exp_name="HGSOC_PorpoiseMMF_OV_ViT_Mayo_metastatic_15_epocs"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/Mayo_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_OV_ViT_UAB_metastatic_15_epocs"

CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/UAB_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_OV_ViT_FHCRC_metastatic_15_epocs"

CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/FHCRC_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/Ov_ViT --omic_sizes [1,2,33,14,14,89] &


### Ctrans path 


exp_name="HGSOC_PorpoiseMMF_CTransPath_Mayo_primary_15_epocs"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/Mayo_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [1,2,33,14,14,89]  &

exp_name="HGSOC_PorpoiseMMF_CTransPath_UAB_primary_15_epocs"

CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/UAB_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [1,2,33,14,14,89]  &

exp_name="HGSOC_PorpoiseMMF_CTransPath_FHCRC_primary_15_epocs"

CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/FHCRC_primary --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [1,2,33,14,14,89]  &


### Metastatic runs

exp_name="HGSOC_PorpoiseMMF_CTransPath_Mayo_metastatic_15_epocs"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/Mayo_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_CTransPath_UAB_metastatic_15_epocs"

CUDA_VISIBLE_DEVICES=1 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/UAB_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [1,2,33,14,14,89] &

exp_name="HGSOC_PorpoiseMMF_CTransPath_FHCRC_metastatic_15_epocs"

CUDA_VISIBLE_DEVICES=2 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/FHCRC_metastatic --exp_code $exp_name --log_data --model_type PorpoiseMMF --max_epoch 15 --weighted_sample --k 5 --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [1,2,33,14,14,89] &

