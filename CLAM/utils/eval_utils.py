import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
from models.multimodal import Multimodal
from models.model_porpoise import PorpoiseMMF
from models.model_coattn import MCAT_Surv
from models.model_SurvPath import SurvPath

import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import ast 

def initiate_model(args, ckpt_path):
    print('Init Model')    
    omic_sizes = ast.literal_eval(args.omic_sizes)

    model_dict = {"omic_sizes": omic_sizes,"dropout": args.drop_out, 'n_classes': args.n_classes, 'embed_dim': args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'multimodal':
        model = Multimodal(**model_dict)
    elif args.model_type == 'PorpoiseMMF':
        model = PorpoiseMMF(**model_dict)
    elif args.model_type == 'MCAT_Surv':
        model = MCAT_Surv(**model_dict)
    elif args.model_type == 'SurvPath':
        model = SurvPath(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    print(ckpt_path)
    ckpt = torch.load(ckpt_path)
    
    ckpt_clean = {}
    # for key in ckpt.keys():
    #     if 'instance_loss_fn' in key:
    #         del ckpt[key]
    #     #     continue
    #     # ckpt_clean.update({key.replace('.module', ''):ckpt[key]})

    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    
    model.load_state_dict(ckpt_clean, strict=True)

    try:
        model.relocate()
    except:
        model.to(device)
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def to_device(data, device):
    """
    Move tensor(s) to a specified device. 
    If data is a list of tensors, each tensor is moved to the device.
    """
    if isinstance(data, list):
        return [item.to(device) for item in data[0]] #data0? is this a bug?....check check ... 
    
    return data.to(device)

def flip_tensor(tensor):
    # Assuming the tensor contains only 0s and 1s
    return 1 - tensor
    
def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, genomic_features, label) in enumerate(loader):
        data, genomic_features,label = data.to(device), to_device(genomic_features, device) ,label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            h = [data,genomic_features]
            logits, Y_prob, Y_hat, _, results_dict = model(h)

        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    print(len(slide_ids)) #only 158 preds made...for 348 slide ids...?...
    print(len(all_labels)) 
    print(len(all_preds))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})

    df = pd.DataFrame(results_dict)
    print(df.head()) #issue!
    return patient_results, test_error, auc_score, df, acc_logger
