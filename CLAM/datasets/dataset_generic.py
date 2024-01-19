from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

def can_convert_to_float(df, column):
    try:
        df[column].astype(float)
        return True
    except ValueError:
        return False



def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		omics_structure="PPI_network_clusters",
		embed_dim = 384,
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		inference = False
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.inference = inference
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col
		self.omics_structure = omics_structure
		self.embed_dim = embed_dim
		slide_data = pd.read_csv(csv_path, dtype=object)
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col, self.inference)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			print(slide_data)
			np.random.shuffle(slide_data.values)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

			

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

		# store ids corresponding each class at the slide level
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col, inference):

		if inference:
			data['label'] = None

		elif label_col != 'label':
			data['label'] = data[label_col].copy()


		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]
		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, omics_structure=self.omics_structure,embed_dim=self.embed_dim)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes,concat_omics=self.concat_omics,embed_dim=self.embed_dim)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):


		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes,concat_omics=self.concat_omics,embed_dim=self.embed_dim)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes,concat_omics=self.concat_omics,embed_dim=self.embed_dim)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes,concat_omics=self.concat_omics, embed_dim=self.embed_dimm)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path 
			print('csv_path', csv_path)
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def load_patho_features(self,data_dirs, slide_id):

		if isinstance(data_dirs, str):
			data_dirs = [data_dirs]
	
		for data_dir in data_dirs:
			full_path = os.path.join(data_dir, 'pt_files', f'{slide_id}.pt')
			if os.path.exists(full_path):
				patho_features = torch.load(full_path)
				return patho_features  # Return immediately after successful load
			
		print("Could not find embedding")
		return torch.rand(2000, self.embed_dim)

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		# Load patho feats.
		patho_features = self.load_patho_features(data_dir, slide_id) 

		if self.omics_structure == "concat":
			### Load the omics features as a vector for a fusion model.
			# genomic_features = ["Signature.1","Signature.2","Signature.3","Signature.5","Signature.8","Signature.13","Microhomology2","Microhomology2ratio","Del/ins-ratio","Del10-ratio","HRD-LOH","Telomeric.AI","LST","DBS2","DBS4","DBS5","DBS6","DBS9","SBS1","SBS2","SBS3","SBS5","SBS8","SBS13","SBS18","SBS26","SBS35","SBS38","SBS39","SBS40","SBS41","ID1","ID2","ID4","ID8"]
			genomic_features_60 = ['RAB25', 'BCL2L1', 'HADH', 'NFKB2', 'COX7A2', 'COX7C', 'TPMT', 'GOLPH3L', 'LTA4H', 'COX6C', 'IDH1', 'YWHAG', 'S100A10', 'COX6A1', 'NDUFB3', 'TGM2', 'CDKN1B', 'NFKB1', 'CAMK2D', 'IL4I1', 'FDX1', 'VCAM1', 'ATM', 'NCAPH2', 'ABCB8', 'IDI1', 'PLIN2', 'ATP6V1D', 'GPX4', 'CA2', 'RELA', 'GLUD1', 'TOP3B', 'RPS6KB2', 'KEAP1', 'LGALS1', 'MTDH', 'AIFM1', 'RHOA', 'CASP7', 'PTGES2', 'TFRC', 'CHUK', 'GPX1', 'PDK1', 'STAT3', 'PECR', 'TALDO1', 'XIAP', 'ACADSB', 'CPOX', 'ARNT', 'BIRC2', 'ACOT7', 'HACL1', 'MYD88', 'EGFR', 'RIPK1', 'NBN', 'LDHA']
			
			# genomic_features = ['ABCB8', 'ACADSB', 'ACOT7', 'AIFM1', 'ARNT', 'ATM',
			# 'ATP6V1D', 'BCL2L1', 'BIRC2', 'CA2', 'CAMK2D', 'CARMIL1', 'CASP7',
			# 'CCDC167', 'CDKN1A', 'CDKN1B', 'CHUK', 'COX6A1', 'COX6C', 'COX7A2',
			# 'COX7C', 'CPOX', 'EGFR', 'FDX1', 'GLUD1', 'GOLPH3L', 'GPX1', 'GPX4',
			# 'HACL1', 'HADH', 'IDH1', 'IDI1', 'IL4I1', 'KEAP1', 'LDHA', 'LGALS1',
			# 'LTA4H', 'MTDH', 'MYD88', 'NBN', 'NCAPH2', 'NDUFB3', 'NFKB1', 'NFKB2',
			# 'PDK1', 'PECR', 'PLIN2', 'PTGES2', 'RAB25', 'RELA', 'RHOA', 'RIPK1',
			# 'RPS6KB2', 'S100A10', 'SENP1', 'STAT3', 'TALDO1', 'TFRC', 'TGM2',
			# 'TOP3B', 'TPMT', 'VCAM1', 'XIAP', 'YWHAG']
			
			# phospo_prots = pd.read_excel("/mnt/ncshare/ozkilim/BRCA/data/HGSOC_processed_data/mmc3.xlsx",sheet_name="Phospho_predictors")
			# phospho_features = phospo_prots["Phospho predictors"].to_list()
			# genomic_features.extend(phospho_features)

			row_data = self.slide_data.loc[idx, genomic_features_60].astype(float)

			# row_data = self.slide_data.iloc[idx, 0:8800].astype(float)
			# Convert to PyTorch tensor
			omics_features = torch.tensor(row_data.values, dtype=torch.float32) 

		elif self.omics_structure == "chowdry_clusters":

			# self.groupings = pd.read_csv("/mnt/ncshare/ozkilim/BRCA/data/HGSOC_processed_data/protien_GO_groupings.csv") #why not in init?....
			omics_features=[]
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
				sub_df = self.slide_data[selected_prots]
				row_data = sub_df.loc[idx,:].astype(float) # get row for protien group
				row_data = torch.tensor(row_data.values, dtype=torch.float32) 
				omics_features.append(row_data) 

			# print("omics shape")
			# print([i.shape[0] for i in omics_features])
		
		elif self.omics_structure == "60_chowdry_clusters":

			omics_features=[]

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
				sub_df = self.slide_data[selected_prots]
				row_data = sub_df.loc[idx,:].astype(float) # get row for protien group
				row_data = torch.tensor(row_data.values, dtype=torch.float32) 
				omics_features.append(row_data) 

			# print("omics shape")
			# print([i.shape[0] for i in omics_features])


		elif self.omics_structure == "PPI_network_clusters":
			omics_features = []
			PPI_clusters = pd.read_csv("/mnt/ncshare/ozkilim/BRCA/data/HGSOC_processed_data/all_shared_proteins_PPI_clusters.csv")
			num_clusters = len(PPI_clusters["Cluster"].value_counts())
			for i in range(num_clusters):
				selected_prots = PPI_clusters[PPI_clusters["Cluster"]==i]
				selected_prots = selected_prots["Protein"].to_list()
				sub_df = self.slide_data[selected_prots]
				row_data = sub_df.loc[idx,selected_prots].astype(float)
				row_data = torch.tensor(row_data.values, dtype=torch.float32) 
				omics_features.append(row_data) 
			
			# print("omics shape")
			# print([i.shape[0] for i in omics_features])

		else: 
	
			omics_features = torch.zeros(60, dtype=torch.float32)

		return patho_features, omics_features, label	
				


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, omics_structure, embed_dim, data_dir=None, num_classes=2):
		self.use_h5 = False
		self.slide_data = slide_data
		self.embed_dim = embed_dim
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
		self.omics_structure = omics_structure

	def __len__(self):
		return len(self.slide_data)
		


