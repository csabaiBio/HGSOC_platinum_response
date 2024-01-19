## Inference for unseen samples


# For all model need to eval....



# all baselines wsis  ViT

CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_ViT_UAB_primary_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_ViT_UAB_primary --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &

CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_ViT_FHCRC_primary_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_ViT_FHCRC_primary --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &

CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_ViT_Mayo_primary_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_ViT_Mayo_primary --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &



CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_ViT_UAB_metastatic_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_ViT_UAB_metastatic --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &

CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_ViT_FHCRC_metastatic_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_ViT_FHCRC_metastatic --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &

CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_ViT_Mayo_metastatic_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_ViT_Mayo_metastatic --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &


# all baselines wsis  OV_ViT



CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_OV_ViT_UAB_primary_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_OV_ViT_UAB_primary --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &

CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_OV_ViT_FHCRC_primary_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_OV_ViT_FHCRC_primary --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &

CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_OV_ViT_Mayo_primary_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_OV_ViT_Mayo_primary --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &



CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_OV_ViT_UAB_metastatic_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_OV_ViT_UAB_metastatic --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &

CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_OV_ViT_FHCRC_metastatic_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_OV_ViT_FHCRC_metastatic --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &

CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_OV_ViT_Mayo_metastatic_15_epocs_PPI_60_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_OV_ViT_Mayo_metastatic --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT --omics_structure PPI_network_clusters  --omic_sizes [27,4,3,5,2,2,4,2,2] &



# # CTransPath (need to make embeddings to test. Also multomdals wont work here!)


# CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_CTransPath_UAB_primary_15_epocs_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_CTransPath_UAB_primary --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [27,4,3,5,2,2,4,2,2]

# CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_CTransPath_FHCRC_primary_15_epocs_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_CTransPath_FHCRC_primary --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [27,4,3,5,2,2,4,2,2]

# CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_CTransPath_Mayo_primary_15_epocs_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_CTransPath_Mayo_primary --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [27,4,3,5,2,2,4,2,2]



# CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_CTransPath_UAB_metastatic_15_epocs_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_CTransPath_UAB_metastatic --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [27,4,3,5,2,2,4,2,2]

# CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_CTransPath_FHCRC_metastatic_15_epocs_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_CTransPath_FHCRC_metastatic --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [27,4,3,5,2,2,4,2,2]

# CUDA_VISIBLE_DEVICES=0 python CLAM/eval.py --models_exp_code /mnt/ncshare/ozkilim/BRCA/results_HGSOC_multimodal/HGSOC_SurvPath_CTransPath_Mayo_metastatic_15_epocs_s1 --save_exp_code TCGA_OV_platinum_multimodal_SurvPath_CTransPath_Mayo_metastatic --task TCGA_proteomic_Ov --model_type SurvPath --results_dir results_HGSOC_multimodal --split all --embeddings_path /tank/WSI_data/Ovarian_WSIs/HGSOC_ovarian/CLAM/level_0/CTransPath --embed_dim 768 --omic_sizes [27,4,3,5,2,2,4,2,2]

