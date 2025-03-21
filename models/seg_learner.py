""" Learner for Few-shot 3D Point Cloud Semantic Segmentation
"""
import torch
from torch import optim
from torch.nn import functional as F

from models.HAFE_Net import HAFENet


class Learner(object):
    def __init__(self, args):
        # init model and optimizer
        self.model = HAFENet(args)
        print('HAFENet and setting optimizer : ')
        self.optimizer = torch.optim.AdamW([
            {"params": self.model.encoder.layers[3:].parameters(), "lr": 1e-4},  
            {"params": self.model.encoder.layers[0:3].parameters(), "lr": 1e-5},    
            {"params": self.model.mlp.parameters(), "lr": 1e-4},
            {"params": self.model.norm.parameters(), "lr": 1e-4},
            #{"params": self.model.qryenh.parameters(), "lr": 1e-4},        
        ], weight_decay=0.1)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=args.gamma)
        self.model.cuda()
        
    def train(self, data):
        """
        Args:
            data: a list of torch tensors wit the following entries.
            - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            - query_x: query point clouds with shape (n_queries, in_channels, num_points)
            - query_y: query labels with shape (n_queries, num_points)
        """
        self.model.train()
        #support_text
        [support_x, support_y, query_x, support_text_prompts, query_y] = data

        # print(data)
        # print(len(data))
        query_logits, loss = self.model(support_x, support_y, query_x, query_y, support_text_prompts)
        
        self.optimizer.zero_grad()
        loss.backward()

        
        self.optimizer.step()
        self.lr_scheduler.step()
        
        query_pred = F.softmax(query_logits, dim=-1).argmax(dim=-1)
        
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0] * query_y.shape[1])

        return loss, accuracy

    def test(self, data):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
            support_text_prompts: support set class label text prompts
        """
        self.model.eval()
        #不需要标签
        [support_x, support_y, query_x, support_text_prompts, query_y] = data
        
        with torch.no_grad():
            logits, _ = self.model(support_x, support_y, query_x, query_y, support_text_prompts)
            
            pred = F.softmax(logits, dim=-1).argmax(dim=-1)
            
            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0] * query_y.shape[1])

        return pred, accuracy
    
    