
# Code for the manuscript "Histopathology and proteomics are synergistic for High-Grade Serous Ovarian Cancer platinum response prediction"

#### Oz Kilim, Alex Olar, András Biricz, Lilla Madaras, Péter Pollner, Zoltán Szállási, Zsofia Sztupinszki, István Csabai

##### "Patients with High-Grade Serous Ovarian Cancer (HGSOC) exhibit varied responses to treatment, with 20-30% showing inherent resistance to platinum-based chemotherapy. While hematoxylin-eosin (H&E) pathological slides are used for routine diagnosis of cancer type, they may also contain diagnostically useful information about treatment response. Our study demonstrates that combining H&E-stained Whole Slide Images (WSIs) with proteomic signatures using a multimodal deep learning framework significantly improves the prediction of platinum response in both discovery and validation cohorts. This method outperforms the Homologous Recombination Deficiency (HRD) score in predicting platinum response and overall patient survival. The study sets new performance benchmarks and explores the intersection of histology and proteomics, highlighting phenotypes related to treatment response pathways, including homologous recombination, DNA damage response, nucleotide synthesis, apoptosis, and ER stress. This integrative approach has the potential to enhance personalized treatment and provide insights into the therapeutic vulnerabilities of HGSOC."
![main](./figures/graphical_abstract_.png)

## 1. Download data, cleaning, and creation of splits.

### Required data for TCGA cohort:

##### WSIs
- GDC data portal TCGA-OV https://portal.gdc.cancer.gov/projects/TCGA-OV

##### Clinical + proteomics 
- mmc2.xlsx (https://www.sciencedirect.com/science/article/pii/S0092867416306730)
- mmc3-2.xlsx (https://www.sciencedirect.com/science/article/pii/S0092867416306730)
- 1-s2.0-S0092867416306730-mmc5.xlsx (https://www.sciencedirect.com/science/article/pii/S0092867416306730)

### Required data for HGSOC cohort:

##### WSIs
- TCIA (https://www.cancerimagingarchive.net/collection/ptrc-hgsoc/)

##### Clinical + proteomics 
- FFPE_discovery_globalprotein_imputed.tsv (https://www.dropbox.com/s/7zul3j1vyrxo40c/processed_data.zip?e=1&dl=0)
- PTRC-HGSOC_List_clinical_data.xlsx (https://www.cancerimagingarchive.net/collection/ptrc-hgsoc/)

#### Setup, pre-processing, and splits: 

```python3 HGSOC_platinum_response/HGSOC_TCGA_tasks_data_setup.py```

## 2. Running classical proteomics-only models.

```python3 HGSOC_platinum_response/classical_models_omics.py```

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
UNI/
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

## 4. Running WSI and Multimodal models. This will use embeddings and tasks to train and test all models and all downstream tasks. 

### Splits are found here: 

```HGSOC_platinum_response/splits```

#### *Ensure all paths are pointing to the correct directories where the generated embeddings exist.

```run.sh```

## 5. Plotting results.

```notebooks/results_analysis/TCGA_HGSOC_results.ipynb```

## 6. Heatmap generation.

```./heatmap.sh```

```notebooks/interpretability/vis_multi_heatmaps.ipynb```

## 7. Histology-proteomics vs genetics tests.

```HGSOC_platinum_response/HRD_results_analysis.ipynb```

## 8. Self-supervised learning pre-training

## Collated data for SSL

| Dataset | Num WSIs | Micron per pixel | Data type |
|----------|----------|----------|----------|
| TCGA_OV | 1481 | 0.5040 | .svs |
| CPTAC_OV | 221 | 0.2501 | .svs |
| HGSC | 349 | 0.4965 | .svs |
| Ovarian Bevacizumab Response | 284 | 0.5 (20X) | .tif |
| UBC OCEAN | 538 | 0.5 | .png |
| Internal | 42 | 0.424 | .mrxs |

### Generate h5 files for all ovarian slides available at correct magnifications.

```./bulk_ovarian_patching.sh```

### Follow instructions at https://github.com/facebookresearch/dino for DINO SSL of ViT.
