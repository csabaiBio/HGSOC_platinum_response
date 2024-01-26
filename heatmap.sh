# Create unimodal heatmaps
CUDA_VISIBLE_DEVICES=0 python CLAM/create_heatmaps_factorized.py --config HGSOC_SurvPath.yaml

# Create multimodal heatmaps

CUDA_VISIBLE_DEVICES=0 python CLAM/create_heatmaps_factorized.py --config HGSOC_SurvPath.yaml


