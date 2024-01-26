# Multimodal High-Grade Serous Ovarian Cancer platinum response prediction

#### Oz Kilim, Alex Olar, András Biricz, Péter Pollner, Zsofia Sztupinszki, Zoltán Szállási, István Csabai

![main](./figures/fig_1_HGSOC.png)

#### This repo contains full code for:

1. Pre-processing and all experiments for the paper: "Histo-proteo multimodal deep learning for High-Grade Serous Ovarian Cancer platinum response prediction"
2. SSL of large scale opensource Ovarian Histopathology WSIs.

## 1. Download, data cleaning cleaning and creation of splits.

```python3 HGSOC_platinum_responce/HGSOC_TCGA_tasks_data_setup.py```

## 2. Running classical Proteomics only models.

```python3 HGSOC_platinum_responce/classical_models_omics.py```

## 3. Patching and embedding data

```setup_general.sh```


Or follow the instructions outlined in https://github.com/mahmoodlab/CLAM to create embeddings for models. Embeddings created should be in the form to follow our analysis with the embedders used:

```bash
CTransPath/
    ├── h5_files
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...
    └── pt_files
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
Lunit-Dino/
    ├── h5_files
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...
    └── pt_files
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
OV-Dino/
    ├── h5_files
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...
    └── pt_files
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
```

## 4. Running WSI and Multimodal models. This will run parts 1 2 and 3. 

#### Ensure all paths are pointing to the correct directories where the generated embeddings exist.

```run.sh```

## 4. Plotting results.

```notebooks/results_analysis/TCGA_HGSOC_results.ipynb```

## 5. Heatmap generation. 

```./heatmap.sh```

```notebooks/interpretability/vis_multi_heatmaps.ipynb```

# 6. Self supervised learning pre-training

## Collated data for SSL

| Dataset | Num WSIs | Micron per pixel | data type | Link |
|----------|----------|----------|----------|----------|
| TCGA_OV | 1481 | 0.5040 | .svs | Row 1, Col 5 |
| CPTAC_OV | 221 | 0.2501 | .svs | Row 2, Col 5 |
| HGSC | 349 | 0.4965 | .svs | Row 3, Col 5 |
| Ovarian Bevacizumab Response | 284 | 0.5 (20X) | .tif | Row 4, Col 5 |
| UBC OCEAN | 538 | 0.5 | .png | Row 5, Col 5 |
| Internal | 42 | 0.424 | .mrxs | Row 6, Col 5 |

### Generate h5 files for all ovarian slides avalalble at correct magnifications. 

```./bulk_ovarian_patching.sh```

