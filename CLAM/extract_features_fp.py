import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
from models.ssl_vit import vit_small
# from models.kimianet import load_kimianet
import argparse
from utils.utils import print_network, collate_features, collate_features_filtered
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
import tiffslide
from tqdm import tqdm

from models.Ov_ViT import Ov_vit_small
from models.CTransPath import CTransPath

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def write_patch_to_hd5(file_path,target_image, output_path, wsi, verbose = 0, print_every=20, custom_downsample=1, target_patch_size=-1):
	"""Output a h5 file containing the RGB values of each patch in the bag.

	Args:
		file_path (_type_): _description_
		target_image: for stain norm
		output_path (_type_): _description_
		wsi (_type_): _description_
		verbose (int, optional): _description_. Defaults to 0.
		print_every (int, optional): _description_. Defaults to 20.
		custom_downsample (int, optional): _description_. Defaults to 1.
		target_patch_size (int, optional): _description_. Defaults to -1.
	"""

	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, target_image=target_image,stain_norm=False,pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}

	loader = DataLoader(dataset=dataset, batch_size=1, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			image, _ = batch
			image, _ = image[0] # NOTE: removing the batch dimension

			asset_dict = { 'image': image }
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


def compute_w_loader(file_path, target_image, output_path, wsi, model, custom_transforms,
 	batch_size = 8, verbose = 0, print_every=20,
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		custom_transforms: custom defined transforms
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, target_image=target_image,stain_norm=False,
        custom_transforms=transform,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch) # need to regenerrate all embed with custom pre-proc to stain norm.
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--ssl_model', default=False, action='store_true')
parser.add_argument('--kimianet', default=False, action='store_true')
parser.add_argument('--OV_ViT', default=False, action='store_true')
parser.add_argument('--CTransPath', default=False, action='store_true')
parser.add_argument('--custom_ssl_model', default=False, action='store_true')
parser.add_argument('--store_patches', default=False, action='store_true')
parser.add_argument('--target_image', type=str, default=None, help="path to target image for stain normalization")

args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	
	if args.store_patches:
		os.makedirs(os.path.join(args.feat_dir, 'h5_patches'), exist_ok=True)
 
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
 
	if args.ssl_model:
		model, transform = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
	elif args.kimianet:
		model, transform = load_kimianet()
	elif args.custom_ssl_model:
		raise NotImplementedError
	elif args.OV_ViT:
		model, transform = Ov_vit_small()
	elif args.CTransPath:
		model, transform = CTransPath()
	else:
		model, transform = resnet50_baseline(pretrained=True)
	model = model.to(device)
	
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		bag_base, _ = os.path.splitext(bag_name)
		torch_path = os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt')
  
		if args.store_patches:
			patch_output_path = os.path.join(args.feat_dir, 'h5_patches', bag_name)
  
		time_start = time.time()

		if args.slide_ext ==".tiff":
			wsi = tiffslide.open_slide(slide_file_path) 
		else:
			wsi = openslide.open_slide(slide_file_path)

		if args.store_patches:
			start = time.time()
			if not os.path.exists(patch_output_path):
				patch_output_path = write_patch_to_hd5(h5_file_path,args.target_image, patch_output_path, wsi, 
					verbose = 0, print_every = 20, custom_downsample=args.custom_downsample, 
					target_patch_size=args.target_patch_size)
			else:
				print('\tAlready processed patches for {}'.format(slide_id))
			time_elapsed = time.time() - start
			print('\ncomputing patches for {} took {} s'.format(patch_output_path, time_elapsed))
  
		try: 
			if not os.path.exists(torch_path):
					output_file_path = compute_w_loader(h5_file_path, args.target_image, output_path, wsi,
						custom_transforms=transform,
						model = model, batch_size = args.batch_size, verbose = 0, print_every = 20, 
						custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
					file = h5py.File(output_file_path, "r")

					features = file['features'][:]
					print('features size: ', features.shape)
					print('coordinates size: ', file['coords'].shape)
					features = torch.from_numpy(features)
					torch.save(features, torch_path)
			else:
				print('\tAlready processed features for {}'.format(slide_id))
				time_elapsed = time.time() - time_start
				# print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
				
		except Exception as e:
			print(e)

