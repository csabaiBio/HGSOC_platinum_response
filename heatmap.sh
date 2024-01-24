CUDA_VISIBLE_DEVICES=0 python CLAM/create_heatmaps.py --config task_1_hunCRC_kimianet_categorical.yaml

CUDA_VISIBLE_DEVICES=0 python CLAM/create_heatmaps.py --config hunCRC_vit_Transcriptomics.yaml


CUDA_VISIBLE_DEVICES=0 python CLAM/create_heatmaps_factorized.py --config HGSOC_SurvPath.yaml
