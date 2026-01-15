"""
Decoder：Transformer 解码器
"""
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from modules import PositionwiseFeedForward, ResidualConnection


class DecoderLayer(nn.Module):
    """
    Decoder 的一层
    
    结构：
        x -> Masked Self-Attention -> Add&Norm 
          -> Encoder-Decoder Attention -> Add&Norm 
          -> FeedForward -> Add&Norm
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
        
        # 1. Masked 自注意力（解码器自身）
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 2. Encoder-Decoder 注意力（关注编码器输出）
        self.encoder_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # 3. 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 三个残差连接
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: torch.Tensor = None, encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            self_mask: 自注意力掩码（防止看到未来信息）[batch_size, 1, tgt_len, tgt_len]
            encoder_mask: 编码器掩码（忽略填充位置）[batch_size, 1, 1, src_len]
        
        Returns:
            [batch_size, tgt_len, d_model]
        """
        # 1. Masked 自注意力 + 残差连接 + 层归一化
        # Q=K=V=x，防止看到未来的 token
        x = self.residual1(x, lambda x: self.self_attention(x, x, x, self_mask)[0])
        
        # 2. Encoder-Decoder 注意力 + 残差连接 + 层归一化
        # Q=x（来自解码器），K=V=encoder_output（来自编码器）
        x = self.residual2(x, lambda x: self.encoder_attention(x, encoder_output, encoder_output, encoder_mask)[0])
        
        # 3. 前馈网络 + 残差连接 + 层归一化
        x = self.residual3(x, self.feed_forward)
        
        return x


class Decoder(nn.Module):
    """
    Decoder：由多个 DecoderLayer 堆叠而成
    
    结构：
        x -> DecoderLayer1 -> DecoderLayer2 -> ... -> DecoderLayerN -> output
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            n_layers: Decoder 层数
            dropout: dropout 概率
        """
        super().__init__()
        
        # 创建 N 层 Decoder
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 最终的层归一化
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: torch.Tensor = None, encoder_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            self_mask: 自注意力掩码
            encoder_mask: 编码器掩码
        
        Returns:
            [batch_size, tgt_len, d_model]
        """
        # 依次通过每一层
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, encoder_mask)
        
        # 最终归一化
        x = self.norm(x)
        
        return x
