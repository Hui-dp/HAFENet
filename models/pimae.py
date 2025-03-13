import os
import torch
from torch import Tensor, nn
from functools import partial
import copy

# -------------------- 模型定义部分 --------------------
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class BatchNormDim1Swap(nn.BatchNorm1d):
    def forward(self, x):
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super().forward(x)
        return x.permute(2, 0, 1)

NORM_DICT = {
    'bn': BatchNormDim1Swap,
    'bn1d': nn.BatchNorm1d,
    'id': nn.Identity,
    'ln': nn.LayerNorm,
}

ACTIVATION_DICT = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leakyrelu': partial(nn.LeakyReLU, negative_slope=0.1)
}

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=128,
                 dropout=0.1, activation="relu", norm_name="ln"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = NORM_DICT[norm_name](d_model)
        self.norm2 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = ACTIVATION_DICT[activation]()

    def with_pos_embed(self, tensor, pos: Tensor):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos: Tensor = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=6):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        
    def forward(self, src, pos: Tensor = None):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        return output

# -------------------- 权重加载部分 --------------------
def loadPretrain(model, path, first_depth, second_depth, is_distributed=False):
    model_dict = model.state_dict()
    pretrain = torch.load(path, map_location='cpu')['base_model']
    
    mapping = {}
    total_depth = first_depth + second_depth
    
    # 构建层映射关系
    for i in range(total_depth):
        if i < first_depth:
            prefix = "pc_branch.MAE_encoder.blocks.layers.{}".format(i)
        else:
            prefix = "blocks.layers.{}".format(i - first_depth)
        
        for name in ['self_attn.in_proj_weight', 'self_attn.in_proj_bias',
                     'self_attn.out_proj.weight', 'self_attn.out_proj.bias',
                     'linear1.weight', 'linear1.bias',
                     'linear2.weight', 'linear2.bias',
                     'norm1.weight', 'norm1.bias',
                     'norm2.weight', 'norm2.bias']:
            model_key = f"layers.{i}.{name}"
            pretrain_key = f"{prefix}.{name}"
            if is_distributed:
                pretrain_key = f"module.{pretrain_key}"
            mapping[model_key] = pretrain_key
    
    # 过滤有效参数
    pretrain_dict = {k: v for k, v in pretrain.items() if k in mapping.values()}
    for model_key, pretrain_key in mapping.items():
        if pretrain_key in pretrain_dict:
            model_dict[model_key] = pretrain_dict[pretrain_key]
    
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded {len(pretrain_dict)}/{len(mapping)} parameters from {path}")

def freeze_layers(model, num_freeze):
    print(f"\n🔧 Freezing first {num_freeze} layers | Total layers: {len(model.layers)}")
    for i, layer in enumerate(model.layers):
        layer_name = f"Layer {i}"
        
        # 尝试获取更详细的层名称（如果有）
        if hasattr(layer, 'name'):
            layer_name = f"{layer_name} ({layer.name})"
        elif hasattr(layer, '_get_name'):
            layer_name = f"{layer_name} ({layer._get_name()})"
            
        # 冻结/解冻逻辑
        if i < num_freeze:
            for param in layer.parameters():
                param.requires_grad_(False)
            print(f"❄️ {layer_name} frozen (params require_grad=False)")
        else:
            for param in layer.parameters():
                param.requires_grad_(True)
            print(f"🔥 {layer_name} unfrozen (params require_grad=True)")
            
    print("✅ Freeze operation completed\n")


def get_pimae_encoder(pretrain_path="/data/HAFENet/models/pimae.pth", 
                     num_freeze_layers=0, 
                     num_keep_layers=3):  # 新增参数控制保留层数
    # 配置参数
    class Args:
        enc_nlayers = 6       # 总层数 = 3 (specific) + 3 (joint)
        enc_dim = 256
        enc_ffn_dim = 128
        enc_dropout = 0.1
        enc_nhead = 4
        enc_activation = "relu"
    
    # 创建模型
    encoder_layer = TransformerEncoderLayer(
        d_model=Args.enc_dim,
        nhead=Args.enc_nhead,
        dim_feedforward=Args.enc_ffn_dim,
        dropout=Args.enc_dropout,
        activation=Args.enc_activation
    )
    model = TransformerEncoder(encoder_layer, Args.enc_nlayers)
    
    # 加载预训练权重
    if os.path.exists(pretrain_path):
        try:
            loadPretrain(model, pretrain_path, 
                        first_depth=3, second_depth=3,
                        is_distributed=True)
        except:
            loadPretrain(model, pretrain_path,
                        first_depth=3, second_depth=3,
                        is_distributed=False)
    else:
        print(f"Warning: Pretrain path {pretrain_path} not found, using random init")
    
    # --- 新增部分：重新初始化指定层之后的权重 ---
    def _reinit_layer(layer):
        """自定义层重新初始化函数"""
        for name, param in layer.named_parameters():
            if 'weight' in name:
                # 线性层/注意力层的权重初始化
                if isinstance(param, nn.Linear):
                    nn.init.xavier_uniform_(param)
                elif isinstance(param, nn.LayerNorm):
                    nn.init.constant_(param, 1.0)
                # 注意力层的特殊初始化
                if 'self_attn' in name:
                    if 'in_proj_weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'out_proj.weight' in name:
                        nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    # 重新初始化指定层之后的参数
    print(f"\n♻️ Reinitializing layers after {num_keep_layers}")
    for idx, layer in enumerate(model.layers):
        if idx >= num_keep_layers:
            layer.apply(_reinit_layer)
            print(f"Layer {idx} reinitialized")
    
    # 冻结指定层
    if num_freeze_layers > 0:
        freeze_layers(model, num_freeze_layers)
    
    return model

# -------------------- 测试示例 --------------------
if __name__ == "__main__":
    # 创建模型
    encoder = get_pimae_encoder(num_freeze_layers=3)  # 参数名对应修改
    
    # 测试输入
    batch_size = 2
    num_points = 1024
    feat_dim = 256
    dummy_input = torch.rand(batch_size, num_points, feat_dim)  # (seq_len, batch, features)
    print(f"Input shape: {dummy_input.shape}")
    dummy_input = dummy_input.transpose(0, 1)
    # 前向传播
    output = encoder(dummy_input)
    output = output.transpose(0, 1)
    print(f"Output shape: {output.shape}")
    
    # # 验证参数冻结
    # print("\nParameter freeze status:")
    # total_params = 0
    # frozen_params = 0
    # for name, param in encoder.named_parameters():
    #     total_params += param.numel()
    #     if not param.requires_grad:
    #         frozen_params += param.numel()
    #     print(f"{name[:30]}... Requires grad: {param.requires_grad}")
    
    # print(f"\nFrozen params: {frozen_params/total_params:.1%} ({frozen_params}/{total_params})")