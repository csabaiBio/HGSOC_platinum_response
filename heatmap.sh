# Create unimodal heatmaps
CUDA_VISIBLE_DEVICES=0 python CLAM/create_heatmaps_factorized.py --config HGSOC_clam_sb.yaml &
# Create multimodal heatmaps
CUDA_VISIBLE_DEVICES=0 python CLAM/create_heatmaps_factorized.py --config HGSOC_MCAT.yaml

# CUDA_VISIBLE_DEVICES=0 python CLAM/create_heatmaps_factorized.py --config HGSOC_SurvPath.yaml
