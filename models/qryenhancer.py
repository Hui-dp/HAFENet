import torch
import torch.nn as nn
import torch.nn.functional as F

# 原型特征torch.Size([2, 256])
# 查询特征torch.Size([2, 2048, 256])

#这个在原始结构上准确率能提升百分之3-4
#默认版本
class CrossAttentionEnhancer(nn.Module):
    def __init__(self, prototype_dim=256, query_dim=256, embed_dim=512, num_heads=8):
        super(CrossAttentionEnhancer, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"
        # 线性投影层
        self.query_proj = nn.Linear(prototype_dim, embed_dim)  # 用原型特征生成 Q
        self.key_proj = nn.Linear(query_dim, embed_dim)        # 用查询特征生成 K
        self.value_proj = nn.Linear(query_dim, embed_dim)      # 用查询特征生成 V

        self.out_proj = nn.Linear(embed_dim, query_dim)         # 输出映射到查询特征维度

        self.norm = nn.LayerNorm(query_dim)  # 归一化层
        

    def forward(self, prototype_feat, query_feat):
        B = prototype_feat.shape[0]
        N = query_feat.shape[1]

        # 通过扩展原型特征的维度，使其与查询特征的第二维匹配
        Q = self.query_proj(prototype_feat.unsqueeze(1).repeat(1, N, 1))  # (B, N, embed_dim)

        # 3. 使用映射后的查询特征生成 K 和 V (B, N, embed_dim)
        K = self.key_proj(query_feat)  # (B, N, embed_dim)
        V = self.value_proj(query_feat) # (B, N, embed_dim)

        # 4. 对 Q, K, V 进行多头分割
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)

        # 5. 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, N, N)
        attn = torch.softmax(scores, dim=-1)  # (B, num_heads, N, N)

        # 6. 加权求和
        out = torch.matmul(attn, V)  # (B, num_heads, N, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)  # (B, N, embed_dim)

        # 7. 通过输出线性层恢复成查询特征的维度
        #去掉投影层
        out = self.out_proj(out)  # (B, N, query_dim)

        # 8. 残差连接和归一化
        out = out + query_feat  # (B, N, query_dim)
        out = self.norm(out)    # (B, N, query_dim)

        return out  # 返回增强后的查询特征


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CrossAttentionEnhancer(nn.Module):
#     def __init__(self, prototype_dim=256, query_dim=256, embed_dim=512, num_heads=8):
#         super().__init__()
#         self.num_heads = num_heads
#         self.embed_dim = embed_dim
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

#         # 投影层
#         self.query_proj = nn.Linear(prototype_dim, embed_dim)
#         self.key_proj = nn.Linear(query_dim, embed_dim)
#         self.value_proj = nn.Linear(query_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, query_dim)

#         # 动态门控层
#         self.gate_layer = nn.Sequential(
#             nn.Linear(2*query_dim, query_dim),
#             nn.GELU(),
#             nn.Linear(query_dim, 1),
#             nn.Sigmoid()
#         )

#         # 注意力输出归一化
#         self.attn_norm = nn.LayerNorm(query_dim)

#     def forward(self, prototype_feat, query_feat):
#         """
#         输入:
#             prototype_feat: (B, P_dim)  原型特征
#             query_feat:     (B, N, Q_dim) 查询特征
#         输出:
#             enhanced_feat: (B, N, Q_dim) 增强后的查询特征
#         """
#         B, N, _ = query_feat.shape
        
#         # 生成QKV（使用广播机制优化显存）
#         Q = self.query_proj(prototype_feat).unsqueeze(1)  # (B, 1, E)
#         K = self.key_proj(query_feat)                     # (B, N, E)
#         V = self.value_proj(query_feat)                   # (B, N, E)

#         # 多头处理（优化后的维度变换）
#         Q = Q.view(B, 1, self.num_heads, self.head_dim).expand(-1, N, -1, -1)
#         K = K.view(B, N, self.num_heads, self.head_dim)
#         V = V.view(B, N, self.num_heads, self.head_dim)

#         # 注意力计算（使用优化后的einsum）
#         scores = torch.einsum('b n h d, b m h d -> b h n m', Q, K) / self.head_dim**0.5
#         attn = F.softmax(scores, dim=-1)
#         out = torch.einsum('b h n m, b m h d -> b n h d', attn, V)
#         out = out.reshape(B, N, self.embed_dim)  # (B, N, E)
        
#         # 输出投影 
#         out = self.out_proj(out)  # (B, N, Q_dim)
        
#         # 动态门控残差
#         gate_input = torch.cat([out, query_feat], dim=-1)
#         gate = self.gate_layer(gate_input)  # (B, N, 1)
#         return self.attn_norm(gate * out + (1 - gate) * query_feat)


# model = EnhancedCrossAttn(
#     prototype_dim=256,
#     query_dim=256,
#     embed_dim=512,
#     num_heads=8
# )

# # 模拟输入
# prototype = torch.randn(32, 256)    # 批次大小32，原型特征维度256
# queries = torch.randn(32, 50, 256)  # 32个样本，50个查询位置

# # 前向传播
# enhanced = model(prototype, queries)  # 输出形状 (32, 50, 256)
# print(enhanced.shape)
