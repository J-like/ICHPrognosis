import torch
import torch.nn as nn
import torch.nn.functional as F  # 用于激活函数等功能


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownConv(nn.Module):
    def __init__(self):
        super(DownConv, self).__init__()
        # 3D 卷积用于下采样
        self.conv_downslice = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=(1, 4, 4), padding=1)
        

    def forward(self, x):
        # 下采样：对 3D 数据进行卷积和下采样
        x = self.conv_downslice(x)
        return x


class Encoder3D(nn.Module):
    def __init__(self, n_channels=1, n_filters=16, normalization='none', has_dropout=False):
        super(Encoder3D, self).__init__()
        self.has_dropout = has_dropout
        self.DownSlice = DownConv()
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.block_five_dw = DownsamplingConvBlock(n_filters * 16, n_filters * 32, normalization=normalization)
        self.block_six = ConvBlock(3, n_filters * 32, n_filters * 32, normalization=normalization)
        

    def encoder(self, input):
        input = self.DownSlice(input)
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)
        # print(x1_dw.shape)
        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        # print(x2_dw.shape)
        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        # print(x3_dw.shape)
        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        # print(x4_dw.shape)
        x5 = self.block_five(x4_dw)
        x5_dw = self.block_five_dw(x5)
        # print(x5_dw.shape)
        x6 = self.block_six(x5_dw)
        
        return x6

    def forward(self, image, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        features = self.encoder(image)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return features





class ConvBlock2D(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock2D, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv2d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock2D(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock2D, self).__init__()

        ops = []
        ops.append(nn.Conv2d(n_filters_in, n_filters_out, kernel_size=stride, padding=0, stride=stride))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder2D(nn.Module):
    def __init__(self, n_channels=3, n_filters=16, normalization='none', has_dropout=False):
        super(Encoder2D, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock2D(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock2D(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock2D(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock2D(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock2D(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock2D(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock2D(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock2D(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock2D(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_dw = DownsamplingConvBlock2D(n_filters * 16, n_filters * 32, normalization=normalization)

        self.block_six = ConvBlock2D(3, n_filters * 32, n_filters * 32, normalization=normalization)

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        x5_dw = self.block_five_dw(x5)

        x6 = self.block_six(x5_dw)

        return x3_dw, x6

    def forward(self, image, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        x3_dw, features = self.encoder(image)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return features
    
    

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, attention_dim, num_outputs=64):
        super(AttentionPooling, self).__init__()
        self.linear = nn.Linear(input_dim, attention_dim)
        self.attention_vectors = nn.Parameter(torch.rand(num_outputs, attention_dim))

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.size()
        attention_scores = self.linear(x)  # [B, N, attention_dim]
        # Compute attention weights: [B, num_outputs, N]
        attention_weights = torch.einsum('bnd,md->bmn', attention_scores, self.attention_vectors)  # [B, num_outputs, N]
        attention_weights = F.softmax(attention_weights, dim=-1)  # Softmax over N
        # Compute weighted sum: [B, num_outputs, C]
        output = torch.bmm(attention_weights, x)  # [B, num_outputs, C]
        return output  # Shape: [B, 64, 512]

class EncoderText(nn.Module):
    def __init__(self, input_dim=768, output_dim=512, heads=8, attention_dim=32, num_outputs=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.scale = (output_dim // heads) ** -0.5

        self.qkv = nn.Linear(input_dim, output_dim * 3, bias=False)
        self.att_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(output_dim, output_dim)
        
        # Removed trans1 since we want to keep the feature dimension at 512
        # Modify AttentionPooling to output 64 tokens
        self.attentionPooling = AttentionPooling(output_dim, attention_dim, num_outputs=num_outputs)
        
    def forward(self, x):
        
        B, N, C = x.size()
        # Adjusted qkv projection
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.output_dim // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention mechanism
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Shape: [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.att_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.output_dim)
        x = self.proj(x)

        # Updated attention pooling
        x = self.attentionPooling(x)  # Outputs shape: [B, 64, 512]

        return x  # Shape: [B, 64, 512]


from abc import ABCMeta, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn

class DownConv(nn.Module):
    def __init__(self):
        super(DownConv, self).__init__()
        # 3D 卷积用于下采样
        self.conv_downslice = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=(1, 4, 4), padding=1)
        

    def forward(self, x):
        # 下采样：对 3D 数据进行卷积和下采样
        x = self.conv_downslice(x)
        return x

class BaseModel(nn.Module):
    """
    一个基础模型类，作为项目中所有模型的父类，
    以确保它们都继承自 nn.Module 并具有统一的接口。
    """
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def input_names(self) -> Sequence[str]:
        """定义模型 forward 方法期望的输入参数名称。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_names(self) -> Sequence[str]:
        """定义模型 forward 方法返回的字典中的键名。"""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """所有子类都必须实现前向传播逻辑。"""
        raise NotImplementedError


def conv3d(in_channels, out_channels, kernel_size=3, stride=1):
    """一个辅助函数，用于创建3D卷积层并自动处理填充。"""
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)


class ConvBnReLU(nn.Module):
    """一个包含 卷积-批量归一化-ReLU 的标准模块。"""
    def __init__(
        self, in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    """一个标准的3D残差块。"""
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, stride=1):
        super().__init__()
        self.conv1 = conv3d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = conv3d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ImageEncoder3D(BaseModel):
    def __init__(
        self,
        in_channels: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
    ) -> None:
        super().__init__()
        self.DownSlice = DownConv()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 尺寸减半
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 尺寸减半
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 尺寸减半
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 尺寸除以4

    @property
    def input_names(self) -> Sequence[str]:
        return ("image",)

    @property
    def output_names(self) -> Sequence[str]:
        return ("output_features",)

    def forward(self, image):
        
        image = self.DownSlice(image)
        
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        # print(out.shape)
        out = self.block2(out)
        # print(out.shape)
        out = self.block3(out)
        # print(out.shape)
        out = self.block4(out)
        # print(out.shape)
        return out
import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 假设x的每个输入样本有9个特征
        self.lstm = nn.LSTM(
            input_size=9,          # 每个时间步的特征维度为9
            hidden_size=256,       # 双向输出为768
            num_layers=2, 
            bidirectional=True,
            batch_first=True
        )
        self.pool = nn.AdaptiveAvgPool1d(64)  # 动态池化至64步长

    def forward(self, x):
        # 假设输入形状为 (B, 9)，并且我们视为时间步长为1的序列
        x = x.unsqueeze(1)  # 将输入形状变为 (B, 1, 9)，作为LSTM的输入

        # LSTM处理
        x, _ = self.lstm(x)  # 输出形状 (B, 1, 768)

        # 转置后进行池化
        x = x.permute(0, 2, 1)  # 变为 (B, 768, 1)
        x = self.pool(x)  # 池化，输出形状 (B, 768, 64)

        return x.permute(0, 2, 1)  # 输出形状 (B, 64, 768)

import torch
import torch.nn as nn
import math

# ===================================================================
# 辅助模块 (请将它们与主模块放在同一个文件中)
# ===================================================================

class PositionalEncoding(nn.Module):
    """ 注入位置信息，无参数 """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe.requires_grad = False
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, embedding_dim]
        return self.dropout(x + self.pe[:x.size(0)])


class SEModule(nn.Module):
    """ 动态特征校准模块，参数量极低 """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, S, C) e.g. (B, 64, 512)
        x_permuted = x.permute(0, 2, 1) # (B, C, S) e.g. (B, 512, 64)
        y = self.squeeze(x_permuted).squeeze(-1) # (B, C)
        y = self.excitation(y).unsqueeze(-1) # (B, C, 1)
        return (x_permuted * y).permute(0, 2, 1) # (B, S, C)


# ===================================================================
# 您的最终模块 (保持名称和接口不变, 输出 B, 64, 512)
# ===================================================================

class BiLSTMEncoder2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- 1. 核心BiLSTM设置 (为保证输出512维，hidden_size设置为256) ---
        self.hidden_size = 256
        self.lstm_output_dim = self.hidden_size * 2  # 256 * 2 = 512

        self.lstm = nn.LSTM(
            input_size=9,
            hidden_size=self.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # --- 2. 轻量级创新模块 (自动适应512维) ---
        # 位置编码器
        self.pos_encoder = PositionalEncoding(d_model=self.lstm_output_dim)

        # 动态特征校准模块
        self.se_module = SEModule(self.lstm_output_dim, reduction_ratio=16)

        # 轻量级门控单元和层归一化
        self.gating_layer = nn.Sequential(
            nn.Linear(self.lstm_output_dim, self.lstm_output_dim),
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(self.lstm_output_dim)

    def forward(self, x):
        # --- 输入形状: (B, 9) ---

        # 1. BiLSTM编码
        x = x.unsqueeze(1)               # (B, 1, 9)
        lstm_out, _ = self.lstm(x)       # (B, 1, 512)

        # 2. 序列生成与位置信息注入
        x = lstm_out.repeat(1, 64, 1)    # (B, 64, 512)
        
        # 添加位置编码
        x = x.permute(1, 0, 2)           # (64, B, 512)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)           # (B, 64, 512)

        # 3. 通过创新模块进行特征提炼
        residual = x # 保存用于残差连接

        recalibrated_x = self.se_module(x)
        gate = self.gating_layer(recalibrated_x)
        gated_output = recalibrated_x * gate

        # 4. 残差连接与归一化
        output = self.ln(residual + gated_output)
        
        # --- 输出形状: (B, 64, 512)，与最新要求完全一致 ---
        return output
    
    
    
    
import torch
import torch.nn as nn

class LightweightGPTExpander(nn.Module):
    def __init__(self, input_dim=512, output_seq_len=64, 
                 mlp_ratio=2.0, use_enhancer=True):
        """
        轻量级GPT特征扩展器
        
        通过一个MLP映射网络将单个特征向量高效地扩展为序列，参数量极小。

        参数:
        input_dim: 输入特征维度 (默认: 512)
        output_seq_len: 目标序列长度 (默认: 64)
        mlp_ratio: MLP中间层的扩展比例 (默认: 2.0)
        use_enhancer: 是否使用初始的瓶颈MLP增强器 (默认: True)
        """
        super().__init__()
        self.output_seq_len = output_seq_len
        self.use_enhancer = use_enhancer
        
        # 1. (可选) 特征增强投影
        if self.use_enhancer:
            bottleneck_dim = max(128, int(input_dim / 4)) # 动态设置瓶颈维度
            self.feature_enhancer = nn.Sequential(
                nn.Linear(input_dim, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, input_dim)
            )
        
        # 2. 位置编码
        # 这是模型学习如何区分序列中不同位置的关键
        self.position_emb = nn.Parameter(torch.randn(1, output_seq_len, input_dim) * 0.02)
        
        # 3. 轻量级MLP映射网络 (替代昂贵的MultiheadAttention)
        hidden_dim = int(input_dim * mlp_ratio)
        self.mapping_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 4. 层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        输入: [batch_size, input_dim]
        输出: [batch_size, output_seq_len, input_dim]
        """
        # 步骤1: (可选) 特征增强
        if self.use_enhancer:
            context = self.feature_enhancer(x) + x # 残差连接
        else:
            context = x
            
        # 步骤2: 扩展全局特征并融合位置信息
        # [B, D] -> [B, 1, D] -> [B, L, D]
        expanded_context = context.unsqueeze(1).expand(-1, self.output_seq_len, -1)
        positioned = expanded_context + self.position_emb
        
        # 步骤3: 通过MLP网络进行非线性映射
        # 残差连接 + 层归一化
        normed_positioned = self.norm1(positioned)
        mapped = self.mapping_network(normed_positioned)
        
        # 第二个残差连接和归一化
        output = self.norm2(positioned + mapped)
        
        return output

# --- 对比和参数计算 ---
if __name__ == "__main__":
    print("--- 原始方案: OptimalGPTEncoder ---")
    original_encoder = OptimalGPTEncoder()
    total_params_original = sum(p.numel() for p in original_encoder.parameters())
    print(f"总参数量: {total_params_original:,}")

    print("\n--- 改进方案: LightweightGPTExpander ---")
    lightweight_encoder = LightweightGPTExpander()
    total_params_lightweight = sum(p.numel() for p in lightweight_encoder.parameters())
    print(f"总参数量: {total_params_lightweight:,}")
    
    # 模拟输入并检查输出形状
    input_tensor = torch.randn(4, 512) # batch_size=4
    output_tensor = lightweight_encoder(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_tensor.shape}")

    reduction = (1 - total_params_lightweight / total_params_original) * 100
    print(f"\n参数量减少了约: {reduction:.2f}%")