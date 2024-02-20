
import torch
import numpy as np 
# from x_transformers import CrossAttender

import torch
import torch.nn as nn
from torch import nn
from einops import reduce
import torch.nn.functional as F

# from x_transformers import Encoder
from torch.nn import ReLU

from models.layers.cross_attention import FeedForward, MMAttentionLayer
import pdb

import math
import pandas as pd

def exists(val):
    return val is not None


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


class SurvPath(nn.Module):
    def __init__(
        self, 
        omic_sizes=[1,2,3,4],
        embed_dim=384,
        dropout=0.1,
        n_classes=2,
        wsi_projection_dim=256,
        omic_names = [],
        size_arg='small',
        subtyping=True, 
        k_sample=8,
        instance_loss_fn=nn.CrossEntropyLoss()
        ):
        
        super(SurvPath, self).__init__()

        #---> general props
        self.num_pathways = len(omic_sizes)
        print("num pathways.")
        print(self.num_pathways)
        self.dropout = dropout

        #---> omics preprocessing for captum
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        #---> wsi props
        self.embed_dim = embed_dim 
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.wsi_projection_dim),
        )

        #---> omics props
        self.init_per_path_model(omic_sizes)

        #---> cross attention props
        self.identity = nn.Identity() # use this layer to calculate ig
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathways
        )

        #---> logits props 
        self.n_classes = n_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # when both top and bottom blocks 
        self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.n_classes)
            )
        
    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)    
    
    def forward(self, h):

        wsi = h[0]
        x_omic = h[1] 

        mask = None
        return_attn = True
        
        #---> get pathway embeddings 
        # [print(sig_feat.shape) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)
        # h_omic_bag = torch.stack(h_omic)
        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi).unsqueeze(0)

        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)
        
        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # when both top and bottom block
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1) #---> both branches
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only

        # embedding = torch.mean(mm_embed, dim=1)
        #---> get logits
        logits = self.to_logits(embedding)

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = [attn_pathways, cross_attn_pathways, cross_attn_histology]

        if return_attn:
            return logits, Y_prob, Y_hat, None , results_dict
        else:
            return logits
        
    def captum(self, wsi, pathway1,pathway2,pathway3,pathway4):
        
        mask = None
        return_attn = True
        
        x_omic = [pathway1,pathway1,pathway1,pathway1,pathway1,] #pass all 80 seperately here... 
        #---> get pathway embeddings 
        # [print(sig_feat.shape) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)
        # h_omic_bag = torch.stack(h_omic)
        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi).unsqueeze(0)

        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)
        
        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # when both top and bottom block
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1) #---> both branches
        # embedding = paths_postSA_embed #---> top bloc only
        # embedding = wsi_postSA_embed #---> bottom bloc only

        # embedding = torch.mean(mm_embed, dim=1)
        #---> get logits
        logits = self.to_logits(embedding)

        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = [attn_pathways, cross_attn_pathways, cross_attn_histology]

        if return_attn:
            return logits, Y_prob, Y_hat, None , results_dict
        else:
            return logits
        
    def relocate(self):
        # put all onto GPU!
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))

            self.wsi_net = nn.DataParallel(self.wsi_net, device_ids=device_ids).to('cuda:0')
            self.sig_networks = self.sig_networks.to('cuda:0')
            self.coattn = nn.DataParallel(self.coattn, device_ids=device_ids).to('cuda:0')
            self.path_transformer = nn.DataParallel(self.path_transformer, device_ids=device_ids).to('cuda:0')
            self.path_attention_head = nn.DataParallel(self.path_attention_head, device_ids=device_ids).to('cuda:0')
            self.path_rho = nn.DataParallel(self.path_rho, device_ids=device_ids).to('cuda:0')
            
            self.omic_transformer = nn.DataParallel(self.omic_transformer, device_ids=device_ids).to('cuda:0')
            self.omic_attention_head = nn.DataParallel(self.omic_attention_head, device_ids=device_ids).to('cuda:0')
            self.omic_rho = nn.DataParallel(self.omic_rho, device_ids=device_ids).to('cuda:0')
            

        if self.fusion is not None:
            self.mm = self.mm.to(device)

        self.classifier = self.classifier.to(device)

