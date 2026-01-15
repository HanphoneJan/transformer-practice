"""
Embedding 层：词嵌入和位置嵌入
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置编码：为输入序列添加位置信息
    
    Transformer 本身没有序列概念，需要显式注入位置信息
    使用正弦和余弦函数生成位置编码
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout 概率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 除数项 [d_model/2]
        # 对于偶数索引使用 10000^(2i/d_model)，奇数索引类似
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 填充位置编码
        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model/2]
        
        # 添加 batch 维度 [1, max_len, d_model] 以便后续广播
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer，不是参数，但会被保存
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        
        Returns:
            添加了位置编码的张量
        """
        # x: [batch_size, seq_len, d_model]
        # self.pe: [1, max_len, d_model]
        # 截取与输入序列长度相同的位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    词嵌入：将词索引转换为稠密向量
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 嵌入维度
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 词索引 [batch_size, seq_len]
        
        Returns:
            词嵌入 [batch_size, seq_len, d_model]
        """
        # 嵌入后乘以 sqrt(d_model) 进行缩放
        return self.embedding(x) * math.sqrt(self.d_model)


class TransformerEmbedding(nn.Module):
    """
    完整的嵌入层：词嵌入 + 位置编码
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout 概率
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 词索引 [batch_size, seq_len]
        
        Returns:
            包含位置信息的嵌入 [batch_size, seq_len, d_model]
        """
        # 词嵌入 [batch_size, seq_len, d_model]
        x = self.token_embedding(x)
        # 添加位置编码
        x = self.positional_encoding(x)
        return x
