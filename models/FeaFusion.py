import torch
import torch.nn as nn
from torch.nn import functional as F

class FeatureFusion(nn.Module):
    def __init__(self, dim_a=256, dim_b=512, hidden_dim=512, output_dim=256):
        """
        初始化特征融合模块，先将特征B降维到与特征A相同的维度，然后通过自适应门控机制按比例混合特征A和特征B，最后通过两个全连接层进行融合。
        
        参数:
        - dim_a (int): 特征A的输入维度
        - dim_b (int): 特征B的输入维度
        - hidden_dim (int): 融合模块中第一个全连接层的输出维度
        - output_dim (int): 融合模块中第二个全连接层的输出维度
        """
        super(FeatureFusion, self).__init__()
        # 将特征B降维到dim_a
        self.reduce_b = nn.Linear(dim_b, dim_a)

        # 门控网络，用于生成混合比例
        self.gate = nn.Sequential(
            nn.Linear(dim_a * 2, dim_a),
            nn.ReLU(),
            nn.Linear(dim_a, dim_a),
            nn.Sigmoid()  # 输出值在 [0,1] 之间
        )

    def forward(self, feat_a, feat_b):
        """
        前向传播函数。
        
        参数:
        - feat_a (Tensor): 第一个输入特征，形状为 (batch_size, dim_a)
        - feat_b (Tensor): 第二个输入特征，形状为 (batch_size, dim_b)
        
        返回:
        - Tensor: 融合后的特征，形状为 (batch_size, output_dim)
        """
        # 确保输入为 float32
        feat_a = feat_a.float()
        feat_b = feat_b.float()

        # 降维特征B到dim_a
        reduced_b = self.reduce_b(feat_b)  # [batch_size, dim_a]

        # 拼接特征A和降维后的特征B用于生成门控权重
        gate_input = torch.cat((feat_a, reduced_b), dim=1)  # [batch_size, dim_a * 2]

        # 生成门控权重
        gate_weights = self.gate(gate_input)  # [batch_size, dim_a]

        # 自适应比例混合
        fused = gate_weights * feat_a + (1 - gate_weights) * reduced_b  # [batch_size, dim_a]

        out = fused

        return out



def basic_usage_example():
    # 定义批次大小和特征维度
    batch_size = 4
    proto_dim = 256
    support_text_dim = 512
    common_dim = 256

    # 创建虚拟的特征张量
    feat_a = torch.randn(batch_size, proto_dim)  # [batch_size, proto_dim]
    feat_b = torch.randn(batch_size, support_text_dim)  # [batch_size, support_text_dim]

    # 实例化 FeatureFusion 模块
    fusion_module = FeatureFusion(proto_dim=proto_dim, support_text_dim=support_text_dim, common_dim=common_dim, num_heads=8)

    # 前向传播
    fused_features = fusion_module(feat_a, feat_b)  # [batch_size, common_dim]

    print("Fused Features Shape:", fused_features.shape)  # 应输出: torch.Size([4, 512])
    print("Fused Features:", fused_features)

if __name__ == "__main__":
    basic_usage_example()

