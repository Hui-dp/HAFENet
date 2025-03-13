""" Cuda util function
"""
import torch
# def cast_cuda(input):
#     if type(input) == type([]):
#         for i in range(len(input)):
#             input[i] = cast_cuda(input[i])
#     else:
#         return input.cuda()
#     return input


def cast_cuda(input):
    """递归将嵌套结构中的 PyTorch 张量转移到 CUDA，跳过其他类型"""
    if isinstance(input, list):
        # 列表：逐元素处理并保留列表结构
        return [cast_cuda(elem) for elem in input]
    elif isinstance(input, tuple):
        # 元组：转换为 list 处理后再转回 tuple
        return tuple(cast_cuda(list(input)))
    elif isinstance(input, dict):
        # 字典：递归处理每个键值对
        return {k: cast_cuda(v) for k, v in input.items()}
    elif isinstance(input, torch.Tensor):
        # PyTorch 张量：转移到 CUDA
        return input.cuda()
    else:
        # 其他类型（如 str、np.ndarray）直接返回
        return input