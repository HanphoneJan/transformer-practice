"""
Encoder：Transformer 编码器
"""
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from modules import PositionwiseFeedForward, ResidualConnection


class EncoderLayer(nn.Module):
    """
    Encoder 的一层
    
    结构：
        x -> Self-Attention -> Add&Norm -> FeedForward -> Add&Norm
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout 概率
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 两个残差连接
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 填充掩码 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 1. 自注意力 + 残差连接 + 层归一化
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, mask)[0])
        
        # 2. 前馈网络 + 残差连接 + 层归一化
        x = self.residual2(x, self.feed_forward)
        
        return x


class Encoder(nn.Module):
    """
    Encoder：由多个 EncoderLayer 堆叠而成
    
    结构：
        x -> EncoderLayer1 -> EncoderLayer2 -> ... -> EncoderLayerN -> output
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            n_layers: Encoder 层数
            dropout: dropout 概率
        """
        super().__init__()
        
        # 创建 N 层 Encoder
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 最终的层归一化
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 填充掩码
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 依次通过每一层
        for layer in self.layers:
            x = layer(x, mask)
        
        # 最终归一化
        x = self.norm(x)
        
        return x
