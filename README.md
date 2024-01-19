# Ovarian WSI workshop. 

#### This repo contains full code for:

1. SSL of large scale opensource Ovarian Histopathology WSIs.
2. Downstream training of various Ovarian cancer-related tasks of interest.

## Setup

### Cloning this repository

With out customized (slightly) CLAM module, in order to properly init this repo, you should run:

```
git clone --recurse-submodules https://github.com/csabaibio/BRCA.git
```

This will recusrsively download CLAM into your directory.

### Install depedencies

```
poetry shell
cd topk/
poetry run python -m build
cd ..
poetry install
```

# 1. Self supervised learning pre-training

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

#### Function of this script: 

- Patching of bulk data.
- Saving patch images to h5 files.

TODO Add: tar script and dino instructions here:

# 2. Downstream tasks 

1. BRCA vs WT prediction.
2. Platinum treatment survival prediction.
3. HD-detect classification.
4. BRCA vs Queiscent descision boundry learning for classifying "unknown" Intact cases

## Generating the tasks.

TODO: fully organise Scripts to generate tasks.


## Results Organised here:
https://docs.google.com/spreadsheets/d/1IBodksS7xHwYTG3GIFfsCSejoapck7ep0UpAsKAbNMI/edit?usp=sharing

## Create splits.

```./create_splits.sh```

## Running models.

```./train.sh```

## Evaluating test sets. 


## Plotting results.

```notebooks/results_analysis/```

## Heatmap generation. 
