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
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
import tiffslide

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi, pretrained,
 	batch_size = 8, verbose = 0, print_every=20,  
	custom_downsample=0, target_patch_size=0, ):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features) #might need an augmentation on this to amek slides more natural earth colors.

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'

	image_tensor_list = []
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():
			# save h5 files as batches... for dino... 
			image_tensor_list.append(batch)
			#append to rest of batch... 
	resulting_tensor = torch.cat(image_tensor_list, dim=0)
	all_images = resulting_tensor.cpu().numpy()
	asset_dict = {'imgs': all_images}
	save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode) 

	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=0)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--normalize', default=True)


args = parser.parse_args()

if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	total = len(bags_dataset) #just get 10 to look at.
	for bag_candidate_idx in range(total):
		try:
			# slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
			slide_id = str(bags_dataset[bag_candidate_idx])

			bag_name = slide_id+'.h5'
			h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
			slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
			print('\nprogress: {}/{}'.format(bag_candidate_idx, total))

			output_path = os.path.join(args.save_path + "/" + bag_name)
			time_start = time.time()
			if args.slide_ext == ".tiff":
				wsi = tiffslide.open_slide(slide_file_path) 
			else:
				wsi = openslide.open_slide(slide_file_path) 
			output_file_path = compute_w_loader(h5_file_path, output_path, wsi , args.normalize,
			batch_size = args.batch_size, verbose = 1, print_every = 20,  
			custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		except Exception as e:
			# Catch and print the error
			print(f"An error occurred: {e}")
