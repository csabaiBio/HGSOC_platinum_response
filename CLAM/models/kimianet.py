from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pdb
from torch.utils.data import Dataset, DataLoader
from glob import glob
from skimage import io, transform
import torch.nn.functional as F
from PIL import Image
import pickle	
from torchvision.models.densenet import DenseNet121_Weights

data_transforms = {
	'train': transforms.Compose([
        # transforms.Resize(1000),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
        # transforms.Resize(1000),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

class fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_1 = x
		out_3 = self.fc_4(x)
		return  out_1, out_3

class remapped_model(nn.Module):
    def __init__(self, model):
        super(remapped_model, self).__init__()
        self.model = model
    def forward(self, x):
        x, y = self.model(x)
        return x # return only the first output, which is the feature vector

def load_kimianet():
    model = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
    model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
    num_ftrs = model.classifier.in_features
    
    model_final = fully_connected(model.features, num_ftrs, 30)
    model_final = model_final.to('cuda:0')
    model_final = nn.DataParallel(model_final)
    
    model_final.load_state_dict(torch.load('./data/model_weights/KimiaNetPyTorchWeights.pth'))
    model_final = model_final.module.to('cpu')

    model = remapped_model(model_final)
    
    return model, data_transforms['val']

if __name__ == '__main__':
    net, transforms = load_kimianet()
    
    print('model summary')

    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    
    print('INPUT SHAPE: ', x.shape)
    print('OUTPUT SHAPE: ', y.shape)