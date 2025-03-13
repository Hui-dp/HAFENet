import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder_Seg
from models.qryenhancer import *
from models.pimae import *
from torch.cuda.amp import autocast
from models.randomized_quantization import *

class Permute(nn.Module):
    """维度重排列模块"""
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class HAFENet(nn.Module):
    def __init__(self, args, qry_enh = True, randomized_quantization = True):
        super(HAFENet, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_points = args.pc_npts
        self.qry_enh = qry_enh
       
        self.randomized_quantization = randomized_quantization
        self.encoder_src = Encoder_Seg(input_points=2048, num_stages=3, embed_dim=60, k_neighbors=16, de_neighbors=10,
                                     alpha=1000, beta=30)
        self.encoder = get_pimae_encoder()
        
        if self.randomized_quantization:
            print('Using Randomized Quantization')
            self.random_quan = RandomizedQuantizationAugModule()
        if self.qry_enh:
            print('Using Query Enhancer')
            self.qryenh = CrossAttentionEnhancer()

        self.mlp = nn.Sequential(nn.Linear(900, 512),
                                nn.LayerNorm(512),
                                nn.ReLU(), 
                                nn.Linear(512, 256), 
                                nn.LayerNorm(256),
                                nn.ReLU(),
                                )
        
        self.norm = nn.LayerNorm(256)
        
    def forward(self, support_x, support_y, query_x, query_y, support_text_prompts=None):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        N_way, K_shot, C, PN = support_x.shape
        
        support_x = support_x.float().cuda().permute(0, 1, 3, 2).view(-1, PN, C)
        support_y = support_y.float().cuda().view(N_way, K_shot, PN)
        query_x = query_x.float().cuda().permute(0, 2, 1).view(-1, PN, C)
        query_y = query_y.cuda().view(-1, PN)

        support_x = self.encoder_src(support_x, variant='training')
        support_x = support_x / support_x.norm(dim=1, keepdim=True)
        query_x = self.encoder_src(query_x, variant='training')
        query_x = query_x / query_x.norm(dim=1, keepdim=True)
        support_x = support_x.permute(0,2,1)
        #随机量化增强
        if self.randomized_quantization:
            support_x = self.random_quan(support_x)
        
        query_x = query_x.permute(0,2,1)
        support_x = self.mlp(support_x)
        query_x = self.mlp(query_x)
        support_x = support_x.transpose(0,1)
        query_x = query_x.transpose(0,1)


        support_features = self.encoder(support_x)
        support_features = support_features / support_features.norm(dim=1, keepdim=True)
        query_features = self.encoder(query_x)
        query_features = query_features / query_features.norm(dim=1, keepdim=True)
        support_feat = support_features.transpose(0,1)
        query_feat = query_features.transpose(0,1)

        support = self.norm(support_x.transpose(0,1) + support_feat)
        query_feat = self.norm(query_x.transpose(0,1) + query_feat)
        
        # obtain prototype
        feature_memory_list, label_memory_list = [], []
        support_feat = support_feat.view(N_way, K_shot, PN, -1)
        
        mask_bg = (support_y == 0)
        bg_features = support_feat[mask_bg]
        
        if bg_features.shape[0] < 1:
            bg_features = torch.ones(1, support_feat.shape[-1]).cuda() * 0.1
        else:
            bg_features = bg_features.mean(0).unsqueeze(0)
        feature_memory_list.append(bg_features)
        label_memory_list.append(torch.tensor(0).unsqueeze(0))
        for i in range(N_way):
            mask_fg = (support_y[i] == 1)
            fg_features = support_feat[i, mask_fg]
            fg_features = fg_features.mean(0).unsqueeze(0)
            feature_memory_list.append(fg_features)
            label_memory_list.append(torch.tensor(i+1).unsqueeze(0))

        feature_memory = torch.cat(feature_memory_list, dim=0)
        
        label_memory = torch.cat(label_memory_list, dim=0).cuda()
        label_memory = F.one_hot(label_memory, num_classes=N_way+1)

        feature_memory = feature_memory / torch.norm(feature_memory, dim=-1, keepdim=True)

        if self.qry_enh:
            query_feat = self.qryenh(feature_memory[1:],query_feat)

        sim = [query_feat[i] @ feature_memory.T for i in range(query_feat.shape[0])]
        sim = torch.stack(sim, dim=0)
        logits = sim @ label_memory.float()
        
        loss = F.cross_entropy(logits.reshape(-1, N_way+1), query_y.reshape(-1,).long())
        
        return logits, loss
