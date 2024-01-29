from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.multimodal import Multimodal
from models.model_porpoise import PorpoiseMMF
from models.model_coattn import MCAT_Surv
from models.model_SurvPath import SurvPath
import json
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
					help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
args = parser.parse_args()


def to_device(data, device):
    """
    Move tensor(s) to a specified device. 
    If data is a list of tensors, each tensor is moved to the device.
    """
    if isinstance(data, list):
        return [item.to(device) for item in data]
    
    return data.to(device)

def infer_single_slide(model, patho_feats, genomic_feats, label, reverse_label_dict, k=1):


    patho_feats, genomic_feats = patho_feats.to(device), to_device(genomic_feats, device)
    # pack features for multimodal models.
    features = [patho_feats, genomic_feats] 
    print(patho_feats.shape)
    print(genomic_feats)

    with torch.no_grad():
        if isinstance(model, (CLAM_SB, CLAM_MB, PorpoiseMMF)):
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item()
            A_omic = A.clone()


            if isinstance(model, (CLAM_MB,)):
                A = A[Y_hat]
                A_omic = A.clone()

            A = A.view(-1, 1).cpu().numpy()

        elif isinstance(model, (SurvPath,MCAT_Surv)):

            logits, Y_prob, Y_hat, _, A = model(features)
            Y_hat = Y_hat.item()

        else:
            raise NotImplementedError

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
        
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

        if isinstance(model, (SurvPath)):
            [attn_pathways, cross_attn_pathways, cross_attn_histology] = A # for SurvPath

            print(attn_pathways.shape) #[9,9] pathways to pathways attention.
            print(cross_attn_pathways.shape) #[9,528] for heatmaps
            print(cross_attn_histology.shape) #[528,9]
            A = cross_attn_pathways[0,:].reshape(-1,1).cpu().numpy() #sample first as the attention. Needs to be (N,1)

        elif isinstance(model, (MCAT_Surv)):
            # for MCAT_surv
            A_coattn = A["coattn"] #[1,1,9,528] #Make way to save 9 of these and plto all in one block.
            A_path = A["path"] #[1,9] # what does this represent?...Check in notebook!
            A_omic = A["omic"] #[1,9]
            A = A_coattn[0,0,:,:].cpu().numpy()

        elif isinstance(model, (PorpoiseMMF)):
            
            logits, Y_prob, Y_hat, A_raw, results_dict

            A_omic = A["omic"] #[1,9]
            A = A_coattn[0,0,:,:].cpu().numpy()
            
        
    return ids, preds_str, probs, A, A_omic

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key]
			
			try: 
				val = dtype(val)
			except:
				val = dtype(float(val))

			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict




def load_config(config_file):
    config_path = os.path.join('CLAM/heatmaps/configs', config_file)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def parse_args_into_namespaces(args, config_dict):
    return {
        'patch_args': argparse.Namespace(**config_dict['patching_arguments']),
        'data_args': argparse.Namespace(**config_dict['data_arguments']),
        'model_args': argparse.Namespace(**{**config_dict['model_arguments'], 'n_classes': config_dict['exp_arguments']['n_classes']}),
        'exp_args': argparse.Namespace(**config_dict['exp_arguments']),
        'heatmap_args': argparse.Namespace(**config_dict['heatmap_arguments']),
        'sample_args': argparse.Namespace(**config_dict['sample_arguments'])
    }

def print_config(config_dict):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for sub_key, sub_value in value.items():
                print(f"{sub_key} : {sub_value}")
        else:
            print(f'\n{key} : {value}')

def calculate_patch_and_step_size(patch_args):
    patch_size = (patch_args.patch_size, patch_args.patch_size)
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print(f'patch_size: {patch_size[0]} x {patch_size[1]}, with {patch_args.overlap:.2f} overlap, step size is {step_size[0]} x {step_size[1]}')
    return patch_size, step_size


def initialize_dataframe(data_args, def_seg_params, def_filter_params, def_vis_params, def_patch_params):
    if data_args.process_list is None:
        slides = get_slides_from_data_dir(data_args)
    else:
        slides = pd.read_csv(data_args.process_list, dtype=object)
    
    return initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

def get_slides_from_data_dir(data_args):
    if isinstance(data_args.data_dir, list):
        slides = [slide for data_dir in data_args.data_dir for slide in os.listdir(data_dir)]
    else:
        slides = sorted(os.listdir(data_args.data_dir))
    return [slide for slide in slides if data_args.slide_ext in slide]

def print_data_frame_info(df):
    print(df)
    process_stack = df.copy()
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(total))
    
    return process_stack

def initialize_model(model_args):
    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))

    if model_args.initiate_fn == 'initiate_model':
        return initiate_model(model_args, ckpt_path)
    else:
        raise NotImplementedError

def create_reverse_label_dict(label_dict):
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    return {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}

def setup_directories(exp_args):
    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

def prepare_blocky_wsi_kwargs(patch_args, heatmap_args, patch_size):
    return {
        'top_left': None, 
        'bot_right': None, 
        'patch_size': patch_size, 
        'step_size': patch_size,
        'custom_downsample': patch_args.custom_downsample, 
        'level': patch_args.patch_level, 
        'use_center_shift': heatmap_args.use_center_shift
    }

def process_slide_name_and_label(process_stack, i, data_args, reverse_label_dict):
    slide_name = process_stack.loc[i, 'slide_id']

    if data_args.slide_ext not in slide_name:
        slide_name += data_args.slide_ext
    print('\nprocessing: ', slide_name)  

    try:
        label = process_stack.loc[i, data_args.label_key]
    except KeyError:
        label = 'Unspecified'

    slide_id = slide_name.replace(data_args.slide_ext, '')

    if not isinstance(label, str):
        grouping = reverse_label_dict[label]
    else:
        grouping = label

    return slide_name, label, slide_id, grouping

def setup_slide_directories(exp_args, grouping, slide_id):
    p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
    os.makedirs(p_slide_save_dir, exist_ok=True)
    print('Creating save directory: ', p_slide_save_dir)

    r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping), slide_id)
    os.makedirs(r_slide_save_dir, exist_ok=True)
    print('Creating save directory: ', r_slide_save_dir)

    return p_slide_save_dir, r_slide_save_dir

def determine_slide_path(data_args, slide_name, process_stack, i):
    if isinstance(data_args.data_dir, str):
        slide_path = os.path.join(data_args.data_dir, slide_name)
    elif isinstance(data_args.data_dir, dict):
        data_dir_key = process_stack.loc[i, data_args.data_dir_key]
        slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
    else:
        raise NotImplementedError

    return slide_path

def load_and_update_params(process_stack, i, def_seg_params, def_filter_params, def_vis_params):
    seg_params = def_seg_params.copy()
    filter_params = def_filter_params.copy()
    vis_params = def_vis_params.copy()

    # Assuming load_params is a function that loads and updates parameters
    seg_params = load_params(process_stack.loc[i], seg_params)
    filter_params = load_params(process_stack.loc[i], filter_params)
    vis_params = load_params(process_stack.loc[i], vis_params)

    # Additional processing for specific keys
    # (e.g., 'keep_ids', 'exclude_ids') as in the original code

    return seg_params, filter_params, vis_params

def initialize_wsi_object(slide_path, mask_file, seg_params, filter_params, patch_args):
    # Assuming initialize_wsi is a function that initializes WSI object
    wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
    wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
	
    return wsi_object, wsi_ref_downsample

def load_features(features_path):
    return torch.load(features_path)

def load_genomic_features(process_stack, i, omics_structure):
    '''Pack up genomic features in the correct format for a given model and its omics structure'''
    
    genomic_feats = []

    if omics_structure == "concat":
        ### Load the omics features as a vector for a fusion model.
        # genomic_features = ["Signature.1","Signature.2","Signature.3","Signature.5","Signature.8","Signature.13","Microhomology2","Microhomology2ratio","Del/ins-ratio","Del10-ratio","HRD-LOH","Telomeric.AI","LST","DBS2","DBS4","DBS5","DBS6","DBS9","SBS1","SBS2","SBS3","SBS5","SBS8","SBS13","SBS18","SBS26","SBS35","SBS38","SBS39","SBS40","SBS41","ID1","ID2","ID4","ID8"]
        genomic_features_60 = ['RAB25', 'BCL2L1', 'HADH', 'NFKB2', 'COX7A2', 'COX7C', 'TPMT', 'GOLPH3L', 'LTA4H', 'COX6C', 'IDH1', 'YWHAG', 'S100A10', 'COX6A1', 'NDUFB3', 'TGM2', 'CDKN1B', 'NFKB1', 'CAMK2D', 'IL4I1', 'FDX1', 'VCAM1', 'ATM', 'NCAPH2', 'ABCB8', 'IDI1', 'PLIN2', 'ATP6V1D', 'GPX4', 'CA2', 'RELA', 'GLUD1', 'TOP3B', 'RPS6KB2', 'KEAP1', 'LGALS1', 'MTDH', 'AIFM1', 'RHOA', 'CASP7', 'PTGES2', 'TFRC', 'CHUK', 'GPX1', 'PDK1', 'STAT3', 'PECR', 'TALDO1', 'XIAP', 'ACADSB', 'CPOX', 'ARNT', 'BIRC2', 'ACOT7', 'HACL1', 'MYD88', 'EGFR', 'RIPK1', 'NBN', 'LDHA']
        row_data = process_stack.loc[i, genomic_features_60].astype(float)
        # Convert to PyTorch tensor
        genomic_feats = torch.tensor(row_data.values, dtype=torch.float32) 

    elif omics_structure == "chowdry_clusters":

        # self.groupings = pd.read_csv("/mnt/ncshare/ozkilim/BRCA/data/HGSOC_processed_data/protien_GO_groupings.csv") #why not in init?....
        phospo_prots = pd.read_excel("/mnt/ncshare/ozkilim/BRCA/data/HGSOC_processed_data/mmc3.xlsx",sheet_name="Phospho_predictors")
        phospho_features = phospo_prots["Phospho predictors"].to_list()

        protein_categories = {
            "Drug Metabolism & Biological Oxidation": ["TPMT"],
            "Hemostasis": ["CARMIL1","CCDC167"],
            "Metabolic": [
                "TALDO1", "COX7A2", "LGALS1", "S100A10", "ACADSB", "COX6C", "COX7C", 
                "CA2", "GPX1", "GPX4", "LDHA", "NDUFB3", "ATP6V1D", "ACOT7", "HACL1", 
                "CPOX", "PTGES2", "GLUD1", "COX6A1", "LTA4H", "CASP7", "IL4I1" , "PECR",
                "YWHAG", "IDI1", "AIFM1", "NBN", "HADH", "PLIN2", "FDX1", "NCAPH2", "IDH1", "ABCB8"
            ],
            "Hypoxia": [
                "TGM2", "RAB25", "CDKN1B", "EGFR", "CDKN1A", "RHOA", "NFKB1", 
                "PDK1", "RPS6KB2", "TFRC", "STAT3", "ARNT", "CAMK2D"
            ],
            "NF-kB": [
                "RELA", "ATM", "BCL2L1", "BIRC2", "VCAM1", "NFKB2", "KEAP1", "RIPK1", "MTDH",
                "CHUK", "MYD88", "GOLPH3L", "TOP3B", "XIAP"
            ],
            "phospho": phospho_features

        }
        
        # Create list of vectors for MCAT. 
        for selected_prots in protein_categories.values():
            sub_df = process_stack[selected_prots]
            row_data = sub_df.loc[i,:].astype(float) # get row for protien group
            row_data = torch.tensor(row_data.values, dtype=torch.float32) 
            genomic_feats.append(row_data) 

        # print("omics shape")
        # print([i.shape[0] for i in omics_features])
    
    elif omics_structure == "60_chowdry_clusters":


        protein_categories = {
            "Drug Metabolism & Biological Oxidation": ["TPMT"],
            "Metabolic": [ "TALDO1",'CA2', "COX7A2", "LGALS1", "S100A10", "ACADSB", "COX6C", "COX7C", 
                "GPX1", "GPX4", "LDHA", "NDUFB3", "ATP6V1D", "ACOT7", "HACL1", 
                "CPOX", "PTGES2", "GLUD1", "COX6A1", "LTA4H", "CASP7", "IL4I1" , "PECR",
                "YWHAG", "IDI1", "AIFM1", "NBN", "HADH", "PLIN2", "FDX1", "NCAPH2", "IDH1", "ABCB8"
            ],
            "Hypoxia": [
                "TGM2", "RAB25", "CDKN1B", "EGFR" , "RHOA", "NFKB1", 
                "PDK1", "RPS6KB2", "TFRC", "STAT3", "ARNT", "CAMK2D"
            ],
            "NF-kB": [
                "RELA", "ATM", "BCL2L1", "BIRC2", "VCAM1", "NFKB2", "KEAP1", "RIPK1", "MTDH",
                "CHUK", "MYD88", "GOLPH3L", "TOP3B", "XIAP"
            ]
        }

        # Create list of vectors for MCAT. 
        for selected_prots in protein_categories.values():
            sub_df = process_stack[selected_prots]
            row_data = sub_df.loc[i,selected_prots].astype(float) # get row for protien group
            row_data = torch.tensor(row_data.values, dtype=torch.float32) 
            genomic_feats.append(row_data) 

        # print("omics shape")
        # print([i.shape[0] for i in omics_features])
    
    elif omics_structure == "plat_response_pathways":

        genomic_feats=[]
        with open('HGSOC_platinum_responce/proteomics_combinations.json', 'r') as file:
            protein_sets = json.load(file)

        protein_categories = protein_sets['plat_response_pathways']

        # Create list of vectors for MCAT. 
        for selected_prots in protein_categories.values():
            sub_df = process_stack[selected_prots]
            row_data = sub_df.loc[i,selected_prots].astype(float) # get row for protien group
            row_data = torch.tensor(row_data.values, dtype=torch.float32) 
            genomic_feats.append(row_data) 



    elif omics_structure == "PPI_network_clusters":
        PPI_clusters = pd.read_csv("/mnt/ncshare/ozkilim/BRCA/data/HGSOC_processed_data/60_shared_proteins_PPI_clusters.csv")
        num_clusters = len(PPI_clusters["Cluster"].value_counts())

        for j in range(num_clusters):
            selected_prots = PPI_clusters[PPI_clusters["Cluster"]==j]
            selected_prots = selected_prots["Protein"].to_list()
            sub_df = process_stack[selected_prots]
            row_data = sub_df.loc[i,selected_prots].astype(float)
            row_data = torch.tensor(row_data.values, dtype=torch.float32) 
            genomic_feats.append(row_data) 
        
    else: 
        genomic_feats = torch.zeros(60, dtype=torch.float32)


    return genomic_feats


def save_segmentation_and_features(wsi_object, mask_file, h5_path, attention_scores, block_map_save_path):
    wsi_object.saveSegmentation(mask_file)
    # Assuming 'save_hdf5' is a defined function
    file = h5py.File(h5_path, "r")
    coords = file['coords'][:]
    file.close()
    
    asset_dict = {'attention_scores': attention_scores, 'coords': coords}

    return save_hdf5(block_map_save_path, asset_dict, mode='w')

def save_predictions(process_stack, i, exp_args, Y_hats_str, Y_probs):
    for c in range(exp_args.n_classes):
        process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
        process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

    os.makedirs('heatmaps/results/', exist_ok=True)
    process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)

def sample_patches_and_save_heatmaps(process_stack, i, wsi_object, patch_args, exp_args, sample_args, scores, coords, label, Y_hats,slide_id):
    samples = sample_args.samples
    for sample in samples:
        if sample['sample']:
            tag = "label_{}_pred_{}".format(label, Y_hats[0])
            sample_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
            os.makedirs(sample_save_dir, exist_ok=True)
            print('sampling {}'.format(sample['name']))

            sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                                         score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
            for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))


def generate_and_save_heatmaps(scores, coords, slide_path, wsi_object, heatmap_args, patch_args, vis_patch_size, r_slide_save_dir, p_slide_save_dir, slide_id, top_left, bot_right,map_number):

    # understand in old code what these values should be... 
    print(heatmap_args.vis_level)
    heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, 
                          use_holes=True, binarize=False, vis_level=heatmap_args.vis_level, blank_canvas=False,thresh=-1,
                          patch_size = vis_patch_size, convert_to_percentiles=True)

    heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap_{}.png'.format(slide_id,map_number)))
    del heatmap

    # heatmap_args.vis_level = 1

    heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 
                        'custom_downsample': heatmap_args.custom_downsample}
    if heatmap_args.use_ref_scores:
        heatmap_vis_args['convert_to_percentiles'] = False

    heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
                                                                                       int(heatmap_args.blur), int(heatmap_args.use_ref_scores), 
                                                                                       int(heatmap_args.blank_canvas), float(heatmap_args.alpha), 
                                                                                       int(heatmap_args.vis_level), int(heatmap_args.binarize), 
                                                                                       float(heatmap_args.binary_thresh), heatmap_args.save_ext)

    heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, 
                          **heatmap_vis_args, binarize=heatmap_args.binarize, blank_canvas=heatmap_args.blank_canvas, 
                          thresh=heatmap_args.binary_thresh, patch_size=vis_patch_size, overlap=patch_args.overlap, 
                          top_left=top_left, bot_right=bot_right)
    if heatmap_args.save_ext == 'jpg':
        heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
    else:
        heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))


def save_configuration(exp_args, config_dict):
    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)


# Main function to use the refactored code
def main(args):
    config_dict = load_config(args.config_file)
    config_dict = parse_config_dict(args, config_dict)
    print_config(config_dict)
    namespaces = parse_args_into_namespaces(args, config_dict)
    patch_size, step_size = calculate_patch_and_step_size(namespaces['patch_args'])

    preset = namespaces['data_args'].preset
    
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]


    df = initialize_dataframe(namespaces['data_args'], def_seg_params, def_filter_params, def_vis_params, def_patch_params)

    process_stack = print_data_frame_info(df)
    model = initialize_model(namespaces['model_args'])

    reverse_label_dict = create_reverse_label_dict(namespaces['data_args'].label_dict)

    setup_directories(namespaces['exp_args'])

    blocky_wsi_kwargs = prepare_blocky_wsi_kwargs(namespaces['patch_args'], namespaces['heatmap_args'], patch_size)

    for i in range(len(process_stack)):
        slide_name, label, slide_id, grouping = process_slide_name_and_label(process_stack, i, namespaces['data_args'], reverse_label_dict)
        p_slide_save_dir, r_slide_save_dir = setup_slide_directories(namespaces['exp_args'], grouping, slide_id)
        slide_path = determine_slide_path(namespaces['data_args'], slide_name, process_stack, i)

        mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
        features_path = os.path.join(namespaces['data_args'].feature_path,"pt_files", slide_id+'.pt')
		##### check if pt_features_file exists ######
        h5_path = os.path.join(namespaces['data_args'].feature_path,"h5_files", slide_id+'.h5')
        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        
        seg_params, filter_params, vis_params = load_and_update_params(process_stack, i, def_seg_params, def_filter_params, def_vis_params)
        wsi_object, wsi_ref_downsample = initialize_wsi_object(slide_path, mask_file, seg_params, filter_params, namespaces['patch_args'])
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))

        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)

        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * namespaces['patch_args'].custom_downsample).astype(int))
        patho_feats = load_features(features_path)

        genomic_feats = load_genomic_features(process_stack, i, namespaces['model_args'].omics_structure)
        
        print("genomic feats shape", genomic_feats[0].shape)
        print("histopatho feats shape", patho_feats.shape)

        process_stack.loc[i, 'bag_size'] = len(patho_feats)
        Y_hats, Y_hats_str, Y_probs, A, A_omic = infer_single_slide(model, patho_feats, genomic_feats, label, reverse_label_dict, namespaces['exp_args'].n_classes)
        
        # save attentions of patways genes and proteins in save dir... 

        np.save(r_slide_save_dir+"/omics_attns.npy", A_omic.cpu())

        # loop over all attn heads here.  

        if A.shape[1] == 1: 
            # clam_sb heatmaps
            atttn_map = A.copy()
            map_number = 0
            block_map_save_path = save_segmentation_and_features(wsi_object, mask_file, h5_path, atttn_map, block_map_save_path)
            save_predictions(process_stack, i, namespaces['exp_args'], Y_hats_str, Y_probs)
            file = h5py.File(block_map_save_path, 'r')
            scores = file['attention_scores'][:]
            coords = file['coords'][:]
            file.close()
            sample_patches_and_save_heatmaps(process_stack, i, wsi_object, namespaces['patch_args'], namespaces['exp_args'], namespaces['sample_args'], scores, coords, label, Y_hats, slide_id)
            print('vis_level')
            print(def_vis_params["vis_level"])
            generate_and_save_heatmaps(scores, coords, slide_path, wsi_object, namespaces['heatmap_args'], namespaces['patch_args'], vis_patch_size, r_slide_save_dir, p_slide_save_dir,slide_id, blocky_wsi_kwargs['top_left'], blocky_wsi_kwargs['bot_right'],map_number)

             
        else:  
            for map_number in range(A.shape[0]):
                # for multimodal rewrite code to make both possiboe. 
                atttn_map = A[map_number,:].reshape(-1,1) 

                block_map_save_path = save_segmentation_and_features(wsi_object, mask_file, h5_path, atttn_map, block_map_save_path)
                save_predictions(process_stack, i, namespaces['exp_args'], Y_hats_str, Y_probs)
                file = h5py.File(block_map_save_path, 'r')
                scores = file['attention_scores'][:]
                coords = file['coords'][:]
                file.close()
                # sample_patches_and_save_heatmaps(process_stack, i, wsi_object, namespaces['patch_args'], namespaces['exp_args'], namespaces['sample_args'], scores, coords, label, Y_hats, slide_id)
                print('vis_level')
                print(def_vis_params["vis_level"])
                generate_and_save_heatmaps(scores, coords, slide_path, wsi_object, namespaces['heatmap_args'], namespaces['patch_args'], vis_patch_size, r_slide_save_dir, p_slide_save_dir,slide_id, blocky_wsi_kwargs['top_left'], blocky_wsi_kwargs['bot_right'],map_number)

    save_configuration(namespaces['exp_args'], config_dict)


		
if __name__ == "__main__":
    # Parse the command line arguments
    main(args)


