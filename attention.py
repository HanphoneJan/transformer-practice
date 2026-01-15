"""
注意力机制：自注意力、多头注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力：注意力的核心计算
    
    计算公式：
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: 查询向量 [batch_size, n_heads, seq_len, d_k]
            key: 键向量 [batch_size, n_heads, seq_len, d_k]
            value: 值向量 [batch_size, n_heads, seq_len, d_v]
            mask: 注意力掩码 [batch_size, 1, 1, seq_len] 或 [batch_size, n_heads, seq_len, seq_len]
        
        Returns:
            注意力输出和注意力权重
        """
        d_k = query.size(-1)
        
        # 1. 计算 QK^T
        # scores: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 2. 缩放：除以 sqrt(d_k)
        scores = scores / math.sqrt(d_k)
        
        # 3. 应用掩码（如果有）
        if mask is not None:
            # 将需要屏蔽的位置设为负无穷，使得 softmax 后为 0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax 归一化
        # attention_weights: [batch_size, n_heads, seq_len, seq_len]
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. 与 V 相乘
        # output: [batch_size, n_heads, seq_len, d_v]
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力：让模型能够从不同的表示子空间关注不同位置的信息
    
    核心思想：
        - 将 d_model 分成 h 个头，每个头独立计算注意力
        - 最后将所有头的输出拼接并通过线性层
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: dropout 概率
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # Q, K, V 的线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.w_o = nn.Linear(d_model, d_model)
        
        # 注意力计算
        self.attention = ScaledDotProductAttention()
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入拆分成多个头
        
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            [batch_size, n_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        
        # 重塑并转置
        # [batch_size, seq_len, n_heads, d_k] -> [batch_size, n_heads, seq_len, d_k]
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多头输出合并
        
        Args:
            x: [batch_size, n_heads, seq_len, d_k]
        
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        
        # 转置并重塑
        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, n_heads, d_k]
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, self.d_model)
        return x
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: 注意力掩码
        
        Returns:
            注意力输出 [batch_size, seq_len, d_model]
            注意力权重 [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 1. 线性投影 Q, K, V
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)    # [batch_size, seq_len, d_model]
        V = self.w_v(value)  # [batch_size, seq_len, d_model]
        
        # 2. 拆分成多头
        Q = self.split_heads(Q)  # [batch_size, n_heads, seq_len, d_k]
        K = self.split_heads(K)  # [batch_size, n_heads, seq_len, d_k]
        V = self.split_heads(V)  # [batch_size, n_heads, seq_len, d_k]
        
        # 3. 计算缩放点积注意力
        # 如果 mask 是 2D 的 [seq_len, seq_len]，扩展到 4D
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        x, attention_weights = self.attention(Q, K, V, mask)
        # x: [batch_size, n_heads, seq_len, d_k]
        # attention_weights: [batch_size, n_heads, seq_len, seq_len]
        
        # 4. 合并多头
        x = self.combine_heads(x)  # [batch_size, seq_len, d_model]
        
        # 5. 输出线性投影
        x = self.w_o(x)
        x = self.dropout(x)
        
        return x, attention_weights
