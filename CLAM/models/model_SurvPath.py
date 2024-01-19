
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

        # print(h_omic_bag.shape)
        # print(wsi_embed.shape)

        # run baseline with TCGA in repo and get shapes....

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
        
    def captum(self, omics_0 ,omics_1 ,omics_2 ,omics_3 ,omics_4 ,omics_5 ,omics_6 ,omics_7 ,omics_8 ,omics_9 ,omics_10 ,omics_11 ,omics_12 ,omics_13 ,omics_14 ,omics_15 ,omics_16 ,omics_17 ,omics_18 ,omics_19 ,omics_20 ,omics_21 ,omics_22 ,omics_23 ,omics_24 ,omics_25 ,omics_26 ,omics_27 ,omics_28 ,omics_29 ,omics_30 ,omics_31 ,omics_32 ,omics_33 ,omics_34 ,omics_35 ,omics_36 ,omics_37 ,omics_38 ,omics_39 ,omics_40 ,omics_41 ,omics_42 ,omics_43 ,omics_44 ,omics_45 ,omics_46 ,omics_47 ,omics_48 ,omics_49 ,omics_50 ,omics_51 ,omics_52 ,omics_53 ,omics_54 ,omics_55 ,omics_56 ,omics_57 ,omics_58 ,omics_59 ,omics_60 ,omics_61 ,omics_62 ,omics_63 ,omics_64 ,omics_65 ,omics_66 ,omics_67 ,omics_68 ,omics_69 ,omics_70 ,omics_71 ,omics_72 ,omics_73 ,omics_74 ,omics_75 ,omics_76 ,omics_77 ,omics_78 ,omics_79 ,omics_80 ,omics_81 ,omics_82 ,omics_83 ,omics_84 ,omics_85 ,omics_86 ,omics_87 ,omics_88 ,omics_89 ,omics_90 ,omics_91 ,omics_92 ,omics_93 ,omics_94 ,omics_95 ,omics_96 ,omics_97 ,omics_98 ,omics_99 ,omics_100 ,omics_101 ,omics_102 ,omics_103 ,omics_104 ,omics_105 ,omics_106 ,omics_107 ,omics_108 ,omics_109 ,omics_110 ,omics_111 ,omics_112 ,omics_113 ,omics_114 ,omics_115 ,omics_116 ,omics_117 ,omics_118 ,omics_119 ,omics_120 ,omics_121 ,omics_122 ,omics_123 ,omics_124 ,omics_125 ,omics_126 ,omics_127 ,omics_128 ,omics_129 ,omics_130 ,omics_131 ,omics_132 ,omics_133 ,omics_134 ,omics_135 ,omics_136 ,omics_137 ,omics_138 ,omics_139 ,omics_140 ,omics_141 ,omics_142 ,omics_143 ,omics_144 ,omics_145 ,omics_146 ,omics_147 ,omics_148 ,omics_149 ,omics_150 ,omics_151 ,omics_152 ,omics_153 ,omics_154 ,omics_155 ,omics_156 ,omics_157 ,omics_158 ,omics_159 ,omics_160 ,omics_161 ,omics_162 ,omics_163 ,omics_164 ,omics_165 ,omics_166 ,omics_167 ,omics_168 ,omics_169 ,omics_170 ,omics_171 ,omics_172 ,omics_173 ,omics_174 ,omics_175 ,omics_176 ,omics_177 ,omics_178 ,omics_179 ,omics_180 ,omics_181 ,omics_182 ,omics_183 ,omics_184 ,omics_185 ,omics_186 ,omics_187 ,omics_188 ,omics_189 ,omics_190 ,omics_191 ,omics_192 ,omics_193 ,omics_194 ,omics_195 ,omics_196 ,omics_197 ,omics_198 ,omics_199 ,omics_200 ,omics_201 ,omics_202 ,omics_203 ,omics_204 ,omics_205 ,omics_206 ,omics_207 ,omics_208 ,omics_209 ,omics_210 ,omics_211 ,omics_212 ,omics_213 ,omics_214 ,omics_215 ,omics_216 ,omics_217 ,omics_218 ,omics_219 ,omics_220 ,omics_221 ,omics_222 ,omics_223 ,omics_224 ,omics_225 ,omics_226 ,omics_227 ,omics_228 ,omics_229 ,omics_230 ,omics_231 ,omics_232 ,omics_233 ,omics_234 ,omics_235 ,omics_236 ,omics_237 ,omics_238 ,omics_239 ,omics_240 ,omics_241 ,omics_242 ,omics_243 ,omics_244 ,omics_245 ,omics_246 ,omics_247 ,omics_248 ,omics_249 ,omics_250 ,omics_251 ,omics_252 ,omics_253 ,omics_254 ,omics_255 ,omics_256 ,omics_257 ,omics_258 ,omics_259 ,omics_260 ,omics_261 ,omics_262 ,omics_263 ,omics_264 ,omics_265 ,omics_266 ,omics_267 ,omics_268 ,omics_269 ,omics_270 ,omics_271 ,omics_272 ,omics_273 ,omics_274 ,omics_275 ,omics_276 ,omics_277 ,omics_278 ,omics_279 ,omics_280 ,omics_281 ,omics_282 ,omics_283 ,omics_284 ,omics_285 ,omics_286 ,omics_287 ,omics_288 ,omics_289 ,omics_290 ,omics_291 ,omics_292 ,omics_293 ,omics_294 ,omics_295 ,omics_296 ,omics_297 ,omics_298 ,omics_299 ,omics_300 ,omics_301 ,omics_302 ,omics_303 ,omics_304 ,omics_305 ,omics_306 ,omics_307 ,omics_308 ,omics_309 ,omics_310 ,omics_311 ,omics_312 ,omics_313 ,omics_314 ,omics_315 ,omics_316 ,omics_317 ,omics_318 ,omics_319 ,omics_320 ,omics_321 ,omics_322 ,omics_323 ,omics_324 ,omics_325 ,omics_326 ,omics_327 ,omics_328 ,omics_329 ,omics_330, wsi):
        
        #---> unpack inputs
        mask = None
        return_attn = False
        
        omic_list = [omics_0 ,omics_1 ,omics_2 ,omics_3 ,omics_4 ,omics_5 ,omics_6 ,omics_7 ,omics_8 ,omics_9 ,omics_10 ,omics_11 ,omics_12 ,omics_13 ,omics_14 ,omics_15 ,omics_16 ,omics_17 ,omics_18 ,omics_19 ,omics_20 ,omics_21 ,omics_22 ,omics_23 ,omics_24 ,omics_25 ,omics_26 ,omics_27 ,omics_28 ,omics_29 ,omics_30 ,omics_31 ,omics_32 ,omics_33 ,omics_34 ,omics_35 ,omics_36 ,omics_37 ,omics_38 ,omics_39 ,omics_40 ,omics_41 ,omics_42 ,omics_43 ,omics_44 ,omics_45 ,omics_46 ,omics_47 ,omics_48 ,omics_49 ,omics_50 ,omics_51 ,omics_52 ,omics_53 ,omics_54 ,omics_55 ,omics_56 ,omics_57 ,omics_58 ,omics_59 ,omics_60 ,omics_61 ,omics_62 ,omics_63 ,omics_64 ,omics_65 ,omics_66 ,omics_67 ,omics_68 ,omics_69 ,omics_70 ,omics_71 ,omics_72 ,omics_73 ,omics_74 ,omics_75 ,omics_76 ,omics_77 ,omics_78 ,omics_79 ,omics_80 ,omics_81 ,omics_82 ,omics_83 ,omics_84 ,omics_85 ,omics_86 ,omics_87 ,omics_88 ,omics_89 ,omics_90 ,omics_91 ,omics_92 ,omics_93 ,omics_94 ,omics_95 ,omics_96 ,omics_97 ,omics_98 ,omics_99 ,omics_100 ,omics_101 ,omics_102 ,omics_103 ,omics_104 ,omics_105 ,omics_106 ,omics_107 ,omics_108 ,omics_109 ,omics_110 ,omics_111 ,omics_112 ,omics_113 ,omics_114 ,omics_115 ,omics_116 ,omics_117 ,omics_118 ,omics_119 ,omics_120 ,omics_121 ,omics_122 ,omics_123 ,omics_124 ,omics_125 ,omics_126 ,omics_127 ,omics_128 ,omics_129 ,omics_130 ,omics_131 ,omics_132 ,omics_133 ,omics_134 ,omics_135 ,omics_136 ,omics_137 ,omics_138 ,omics_139 ,omics_140 ,omics_141 ,omics_142 ,omics_143 ,omics_144 ,omics_145 ,omics_146 ,omics_147 ,omics_148 ,omics_149 ,omics_150 ,omics_151 ,omics_152 ,omics_153 ,omics_154 ,omics_155 ,omics_156 ,omics_157 ,omics_158 ,omics_159 ,omics_160 ,omics_161 ,omics_162 ,omics_163 ,omics_164 ,omics_165 ,omics_166 ,omics_167 ,omics_168 ,omics_169 ,omics_170 ,omics_171 ,omics_172 ,omics_173 ,omics_174 ,omics_175 ,omics_176 ,omics_177 ,omics_178 ,omics_179 ,omics_180 ,omics_181 ,omics_182 ,omics_183 ,omics_184 ,omics_185 ,omics_186 ,omics_187 ,omics_188 ,omics_189 ,omics_190 ,omics_191 ,omics_192 ,omics_193 ,omics_194 ,omics_195 ,omics_196 ,omics_197 ,omics_198 ,omics_199 ,omics_200 ,omics_201 ,omics_202 ,omics_203 ,omics_204 ,omics_205 ,omics_206 ,omics_207 ,omics_208 ,omics_209 ,omics_210 ,omics_211 ,omics_212 ,omics_213 ,omics_214 ,omics_215 ,omics_216 ,omics_217 ,omics_218 ,omics_219 ,omics_220 ,omics_221 ,omics_222 ,omics_223 ,omics_224 ,omics_225 ,omics_226 ,omics_227 ,omics_228 ,omics_229 ,omics_230 ,omics_231 ,omics_232 ,omics_233 ,omics_234 ,omics_235 ,omics_236 ,omics_237 ,omics_238 ,omics_239 ,omics_240 ,omics_241 ,omics_242 ,omics_243 ,omics_244 ,omics_245 ,omics_246 ,omics_247 ,omics_248 ,omics_249 ,omics_250 ,omics_251 ,omics_252 ,omics_253 ,omics_254 ,omics_255 ,omics_256 ,omics_257 ,omics_258 ,omics_259 ,omics_260 ,omics_261 ,omics_262 ,omics_263 ,omics_264 ,omics_265 ,omics_266 ,omics_267 ,omics_268 ,omics_269 ,omics_270 ,omics_271 ,omics_272 ,omics_273 ,omics_274 ,omics_275 ,omics_276 ,omics_277 ,omics_278 ,omics_279 ,omics_280 ,omics_281 ,omics_282 ,omics_283 ,omics_284 ,omics_285 ,omics_286 ,omics_287 ,omics_288 ,omics_289 ,omics_290 ,omics_291 ,omics_292 ,omics_293 ,omics_294 ,omics_295 ,omics_296 ,omics_297 ,omics_298 ,omics_299 ,omics_300 ,omics_301 ,omics_302 ,omics_303 ,omics_304 ,omics_305 ,omics_306 ,omics_307 ,omics_308 ,omics_309 ,omics_310 ,omics_311 ,omics_312 ,omics_313 ,omics_314 ,omics_315 ,omics_316 ,omics_317 ,omics_318 ,omics_319 ,omics_320 ,omics_321 ,omics_322 ,omics_323 ,omics_324 ,omics_325 ,omics_326 ,omics_327 ,omics_328 ,omics_329 ,omics_330]

        #---> get pathway embeddings 
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(omic_list)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic, dim=1)  #.unsqueeze(0) ### omic embeddings are stacked (to be used in co-attention)
        
        #---> project wsi to smaller dimension (same as pathway dimension)
        wsi_embed = self.wsi_projection_net(wsi)

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
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)

        #---> get logits
        logits = self.to_logits(embedding)

        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        if return_attn:
            return risk, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return risk
        
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

