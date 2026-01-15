"""
Transformer 模块：Feed Forward, LayerNorm, Residual Connection
"""
import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    位置感知前馈网络：在每个位置独立应用相同的两层神经网络
    
    公式：
        FFN(x) = max(0, xW1 + b1)W2 + b2
    
    结构：
        x -> Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model) -> output
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度（通常是 d_model 的 4 倍）
            dropout: dropout 概率
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 第一层：d_model -> d_ff
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # 第二层：d_ff -> d_model
        x = self.w2(x)
        x = self.dropout(x)
        
        return x


class ResidualConnection(nn.Module):
    """
    残差连接 + 层归一化
    
    结构：
        x -> LayerNorm -> (x + Sublayer(x)) -> dropout
        或
        x -> (x + Sublayer(x)) -> LayerNorm
    
    这里使用 Post-LN 形式：先 Sublayer，后 LayerNorm
    """
    def __init__(self, size: int, dropout: float = 0.1):
        """
        Args:
            size: 特征维度
            dropout: dropout 概率
        """
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, sublayer: torch.nn.Module) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            sublayer: 子层（如注意力层或前馈网络）
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 残差连接：x + sublayer(x)
        return x + self.dropout(sublayer(self.norm(x)))
