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

	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB)):
			logits, Y_prob, Y_hat, A, _ = model(features)
			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

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

	return ids, preds_str, probs, A

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

if __name__ == '__main__':
	config_path = os.path.join('CLAM/heatmaps/configs', args.config_file)
	config_dict = yaml.safe_load(open(config_path, 'r'))
	config_dict = parse_config_dict(args, config_dict)

	for key, value in config_dict.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))

	args = config_dict
	patch_args = argparse.Namespace(**args['patching_arguments'])
	data_args = argparse.Namespace(**args['data_arguments'])
	model_args = args['model_arguments']
	model_args.update({'n_classes': args['exp_arguments']['n_classes']})
	model_args = argparse.Namespace(**model_args)
	exp_args = argparse.Namespace(**args['exp_arguments'])
	heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
	sample_args = argparse.Namespace(**args['sample_arguments'])

	patch_size = tuple([patch_args.patch_size for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

	
	preset = data_args.preset
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


	if data_args.process_list is None:
		if isinstance(data_args.data_dir, list):
			slides = []
			for data_dir in data_args.data_dir:
				slides.extend(os.listdir(data_dir))
		else:
			slides = sorted(os.listdir(data_args.data_dir))
		slides = [slide for slide in slides if data_args.slide_ext in slide]
		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
		
	else:
		df = pd.read_csv(data_args.process_list, dtype=object)
		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	print(df)
	process_stack = df.copy()
	# mask = df['process'] == 0
	# process_stack = df[mask]
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	ckpt_path = model_args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	
	if model_args.initiate_fn == 'initiate_model':
		model =  initiate_model(model_args, ckpt_path)
	else:
		raise NotImplementedError



	label_dict =  data_args.label_dict
	class_labels = list(label_dict.keys())
	class_encodings = list(label_dict.values())
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
	'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

	for i in range(len(process_stack)):
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

		p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)
  
		print('Creating save directory: ', p_slide_save_dir)

		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)
  
		print('Creating save directory: ', r_slide_save_dir)

		if heatmap_args.use_roi:
			x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
			y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
			top_left = (int(x1), int(y1))
			bot_right = (int(x2), int(y2))
		else:
			top_left = None
			bot_right = None
		
		print('slide id: ', slide_id)
		print('top left: ', top_left, ' bot right: ', bot_right)

		if isinstance(data_args.data_dir, str):
			slide_path = os.path.join(data_args.data_dir, slide_name)
		elif isinstance(data_args.data_dir, dict):
			data_dir_key = process_stack.loc[i, data_args.data_dir_key]
			slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
		else:
			raise NotImplementedError

		mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
		
		# Load segmentation and filter parameters
		seg_params = def_seg_params.copy()
		filter_params = def_filter_params.copy()
		vis_params = def_vis_params.copy()

		seg_params = load_params(process_stack.loc[i], seg_params)
		filter_params = load_params(process_stack.loc[i], filter_params)
		vis_params = load_params(process_stack.loc[i], vis_params)

		keep_ids = str(seg_params['keep_ids'])
		if len(keep_ids) > 0 and keep_ids != 'none':
			seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
		else:
			seg_params['keep_ids'] = []

		exclude_ids = str(seg_params['exclude_ids'])
		if len(exclude_ids) > 0 and exclude_ids != 'none':
			seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
		else:
			seg_params['exclude_ids'] = []

		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))

		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))

		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))
		
		print('Initializing WSI object')
		wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
		print('Done!')

		wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

		# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
		vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

		block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
		mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
		if vis_params['vis_level'] < 0:
			best_level = wsi_object.wsi.get_best_level_for_downsample(32)
			vis_params['vis_level'] = best_level
		mask = wsi_object.visWSI(**vis_params, number_contours=True)
		mask.save(mask_path)
		
		features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
		h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
	
		##### check if pt_features_file exists ######
		h5_path = os.path.join(data_args.feature_path,"h5_files", slide_id+'.h5') #addeed .... was not here before.

		# features_path =  #Nofeature path?....?????????????? not even loaded in script.... 
		if not os.path.isfile(features_path): 
			file = h5py.File(h5_path, "r")
			features = torch.tensor(file['features'][:])
			torch.save(features, features_path)
			file.close()

		# load histopathology features 
		patho_feats = torch.load(features_path)
		# load genomic features.

		genomic_feats = []

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
			print(selected_prots)
			sub_df = process_stack[selected_prots]
			row_data = sub_df.loc[i,selected_prots].astype(float) # get row for protien group
			row_data = torch.tensor(row_data.values, dtype=torch.float32) 
			genomic_feats.append(row_data) 

		# PPI_clusters = pd.read_csv("/mnt/ncshare/ozkilim/BRCA/data/HGSOC_processed_data/60_shared_proteins_PPI_clusters.csv")

		# num_clusters = len(PPI_clusters["Cluster"].value_counts())
		# for j in range(num_clusters):
		# 	selected_prots = PPI_clusters[PPI_clusters["Cluster"]==j]
		# 	selected_prots = selected_prots["Protein"].to_list()
		# 	sub_df = process_stack[selected_prots]
		# 	row_data = sub_df.loc[i,selected_prots].astype(float)
		# 	row_data = torch.tensor(row_data.values, dtype=torch.float32) 
		# 	genomic_feats.append(row_data) 

		print("genomic feats shape", genomic_feats[0].shape)
		print("histopatho feats shape", patho_feats.shape)

		process_stack.loc[i, 'bag_size'] = len(patho_feats)
		
		wsi_object.saveSegmentation(mask_file)
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, patho_feats, genomic_feats, label, reverse_label_dict, exp_args.n_classes)


		if isinstance(model, (SurvPath)):
			[attn_pathways, cross_attn_pathways, cross_attn_histology] = A # for SurvPath
			print(attn_pathways.shape) #[9,9]
			print(cross_attn_pathways.shape) #[9,528]
			print(cross_attn_histology.shape) #[528,9]
			A = cross_attn_pathways[0,:].reshape(-1,1).cpu().numpy() #sample first as the attention. Needs to be (N,1)

		elif isinstance(model, (MCAT_Surv)):
			# for MCAT_surv
			A_coattn = A["coattn"] #[1,1,9,528] #Make way to save 9 of these and plto all in one block.
			A_path = A["path"] #[1,9] # what does this represent?...Check in notebook!
			A_omic = A["omic"] #[1,9]
			print(A_coattn.shape)
			print(A_path.shape)
			print(A_omic.shape)
			A = A_coattn[0,0,1,:].reshape(-1,1).cpu().numpy()

		#TODO save this for all 9 groups! , make plotting script to investigate all attns. as well as the other attention and their interpretation!

		# loop over all co-attention layers.... 

		# factorize code into functions... 

		print(A.dtype) #should be float32.

		del patho_feats
		del genomic_feats
		
		# if not os.path.isfile(block_map_save_path): 
		file = h5py.File(h5_path, "r")
		coords = file['coords'][:]
		file.close()
		asset_dict = {'attention_scores': A, 'coords': coords} 
		block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w') 
		
		print('Y_hats: ', Y_hats_str)
  
		# save top 3 predictions
		for c in range(exp_args.n_classes):
			process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
			process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

		os.makedirs('heatmaps/results/', exist_ok=True)
		process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
		
		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		samples = sample_args.samples
		for sample in samples:
			if sample['sample']:
				tag = "label_{}_pred_{}".format(label, Y_hats[0])
				sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
				os.makedirs(sample_save_dir, exist_ok=True)
				print('sampling {}'.format(sample['name']))
				sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
					score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
				for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
					print('coord: {} score: {:.3f}'.format(s_coord, s_score))
					patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
					patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

		wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
		'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

		heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)

		print("here")
		print(scores.shape)
		print(coords.shape)
		print(slide_path)
		print(wsi_object)

		heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
						thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
	
		heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
		del heatmap

		save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

		if heatmap_args.use_ref_scores:
			ref_scores = scores
		else:
			ref_scores = None

		if not os.path.isfile(save_path):
			print('heatmap {} not found'.format(save_path))
			if heatmap_args.use_roi:
				save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
				print('found heatmap for whole slide')
				save_path = save_path_full
			else:
				continue

		file = h5py.File(save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
		if heatmap_args.use_ref_scores:
			heatmap_vis_args['convert_to_percentiles'] = False

		heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
																						int(heatmap_args.blur), 
																						int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
																						float(heatmap_args.alpha), int(heatmap_args.vis_level), 
																						int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
		heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
								cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
								binarize=heatmap_args.binarize, 
								blank_canvas=heatmap_args.blank_canvas,
								thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
								overlap=patch_args.overlap, 
								top_left=top_left, bot_right = bot_right)
		if heatmap_args.save_ext == 'jpg':
			heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
		else:
			heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
		
		if heatmap_args.save_orig:
			if heatmap_args.vis_level >= 0:
				vis_level = heatmap_args.vis_level
			else:
				vis_level = vis_params['vis_level']
			heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
			if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				pass
			else:
				heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
				if heatmap_args.save_ext == 'jpg':
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

	with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)


