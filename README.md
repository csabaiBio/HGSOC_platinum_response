# Multimodal High-Grade Serous Ovarian Cancer platinum response prediction

#### This repo contains full code for:

1. SSL of large scale opensource Ovarian Histopathology WSIs.
2. Downstream training of various Ovarian cancer-related tasks of interest.

## Setup

yaml provided

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



## Cleaning all data and creation of splits.

## Running classical omics only models.

```./train.sh```

## Running clam_sb and Multimodal models.


## Plotting results.

```notebooks/results_analysis/```

## Heatmap generation. 


