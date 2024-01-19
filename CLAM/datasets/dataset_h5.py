from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F
import torchstain
import openslide
import random
import string 
from tiatoolbox.tools import stainnorm

from PIL import Image
import h5py

from random import randrange

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

		trnsfrms_val = transforms.Compose(
						[
						transforms.ToTensor(),
						transforms.Normalize(mean = mean, std = std)
						]
					)
	else: #raw 
		trnsfrms_val = transforms.Compose(
						[
						transforms.ToTensor(),
						]
					)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		target_image,
		stain_norm=False,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.stain_norm=stain_norm
		self.target_image=target_image
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()

		if self.stain_norm:
			target_slide = openslide.OpenSlide(target_image)
			target_image = target_slide.read_region((0, 0), 2, target_slide.level_dimensions[2])
			target_image = np.array(target_image)[:,:,0:3]
			method_name="Macenko"
			self.stain_normalizer = stainnorm.get_normalizer(method_name)
			self.stain_normalizer.fit(target_image)


	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)
	
	def stain_normalize(self,source_image):
		"""This takes in a referance target WSI image and norms patches to this referance"""

		normed_image = self.stain_normalizer.transform(source_image)

		return normed_image



	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		try:
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		except Exception as e:
			img = Image.new('RGB', (224, 224), 'white')
			print(e)

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
			
		if self.stain_norm:
			try:
				img = np.array(img.convert('RGB'))
				img = self.stain_normalize(img) #fails on very white images. Ignore those anyway...
				img = Image.fromarray(img) 
				# random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
				# # Save the image
				# img.save('/mnt/ncshare/ozkilim/BRCA/figures/brca_patches/normed_image'+random_string+'.png')
				
			except:
			# 	### Images could not be transformed as they are some strange/white image . This is filtered out later with the new collate function . 
			# 	# # Create an image from the array
			# 	# img_save = img.astype(np.uint8)
			# 	# image = Image.fromarray(img_save)
			# 	print("failed to norm")
				img=None # reject these... 
			
		img = self.roi_transforms(img).unsqueeze(0)

		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




