__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding']

import torch
from gluonts.mx import activation
from torch import nn
import math

class Transpose(nn.Module):
    """对张量进行维度转置"""
    def __init__(self,*dim,contiguous=True):
        # 控制转置后是否调用 contiguous() 方法确保内存连续。
        super().__init__()
        self.dims = dim
        self.contiguous = contiguous
    def forward(self, x):
        if self.contiguous: #强制内存连续
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

def get_activation_fn(activation):
    """
    用于自定义激活函数
    支持两种输入方式：1.预定义激活函数；2.自定义激活函数
    """
    if callable(activation):
        return activation
    elif activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'gelu':
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')

class moving_avg(nn.Module):
    """
    移动平均
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self,x):
        #进行前、后填充，避免边缘信息丢失
        front = x[:,0:1,:].repeat(1, (self.kernel_size-1) // 2,1)
        end = x[:,-1:,:].repeat(1, (self.kernel_size-1) // 2,1)
        x = torch.cat([front,x,end],dim=1)
        x = self.avg(x.permute(0,2,1))
        x = x.permute(0,2,1)
        return x

class series_decomp(nn.Module):
    """
    趋势分解模块
    """
    def __init__(self,kernel_size):
        super(series_decomp,self).__init__()
        self.moving_avg = moving_avg(kernel_size,stride=1)
    def forward(self,x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res,moving_mean

def PositionalEncoding(q_len, d_model, normalize=True):
    """生成正余弦位置编码矩阵"""
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0,q_len).unsqueeze(1) # 生成位置索引
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # 计算频率因子
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std()* 10)
    return pe
SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    """
    二维坐标位置编码，同时考虑序列位置和特征维度，通过迭代调整指数参数x使编码均值为0，增强对称性
    exponential：
        0：指数参数x=0.5；非线性编码，编码更平滑
        1：指数参数x=1；线性编码
    """
    x = .5 if exponential else 1.0
    i = 0
    for i in range(100): # 迭代调整x使编码均值为0
        cpe = 2 * (torch.linspace(0,1,q_len).reshape(-1,1) ** x ) * (torch.linspace(0,1,d_model).reshape(-1,1) ** x) - 1
        if abs(cpe.mean()) <= eps: # 均值接近0时停止迭代
            break
        elif abs(cpe.mean()) > eps: # 调整x使均值趋近0
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    """一维位置编码"""
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe,learn_pe,q_len,d_model):
    if pe == None:
        W_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)