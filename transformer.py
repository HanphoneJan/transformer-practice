"""
完整的 Transformer 模型
"""
import torch
import torch.nn as nn
from embeddings import TransformerEmbedding
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    
    结构：
        Encoder:
            Input Embedding -> Positional Encoding -> N x Encoder Layer -> Output
        
        Decoder:
            Output Embedding -> Positional Encoding -> N x Decoder Layer -> Linear -> Output
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 n_heads: int = 8, d_ff: int = 2048, n_layers: int = 6,
                 dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            src_vocab_size: 源语言词汇表大小
            tgt_vocab_size: 目标语言词汇表大小
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            n_layers: Encoder/Decoder 层数
            dropout: dropout 概率
            max_len: 最大序列长度
        """
        super().__init__()
        
        # Embedding 层
        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len, dropout)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len, dropout)
        
        # Encoder 和 Decoder
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers, dropout)
        
        # 输出投影层：将 d_model 投影到词汇表大小
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        创建填充掩码
        
        Args:
            seq: 输入序列 [batch_size, seq_len]
            pad_idx: 填充 token 的索引
        
        Returns:
            [batch_size, 1, 1, seq_len]，填充位置为 0，其他为 1
        """
        # pad_idx 的位置为 False（0），其他为 True（1）
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def create_look_ahead_mask(self, seq_len: int) -> torch.Tensor:
        """
        创建前瞻掩码（因果掩码）
        
        在解码时，防止看到未来的信息
        
        Args:
            seq_len: 序列长度
        
        Returns:
            [seq_len, seq_len]，下三角为 1，上三角为 0
        """
        # 创建上三角矩阵（对角线以上为 1，以下为 0）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        # 反转：有效位置为 1，屏蔽位置为 0
        mask = (1 - mask).unsqueeze(0)
        return mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_pad_idx: int = 0, tgt_pad_idx: int = 0) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
            src_pad_idx: 源序列填充 token 索引
            tgt_pad_idx: 目标序列填充 token 索引
        
        Returns:
            输出 logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # 创建编码器的填充掩码
        src_mask = self.create_padding_mask(src, src_pad_idx)
        
        # 创建解码器的掩码
        tgt_mask = self.create_padding_mask(tgt, tgt_pad_idx)  # 填充掩码
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len).to(tgt.device)  # 前瞻掩码
        
        # 合并解码器的两个掩码
        if tgt_mask is not None:
            # look_ahead_mask: [1, tgt_len, tgt_len]
            # tgt_mask: [batch_size, 1, 1, tgt_len]
            # 需要 broadcast 到 [batch_size, 1, tgt_len, tgt_len]
            tgt_mask = tgt_mask & look_ahead_mask.bool()
        else:
            tgt_mask = look_ahead_mask
        
        # Encoder 前向传播
        # src: [batch_size, src_len] -> src_embedded: [batch_size, src_len, d_model]
        src_embedded = self.src_embedding(src)
        encoder_output = self.encoder(src_embedded, src_mask)
        
        # Decoder 前向传播
        # tgt: [batch_size, tgt_len] -> tgt_embedded: [batch_size, tgt_len, d_model]
        tgt_embedded = self.tgt_embedding(tgt)
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        # decoder_output: [batch_size, tgt_len, d_model] -> logits: [batch_size, tgt_len, tgt_vocab_size]
        logits = self.output_projection(decoder_output)
        
        return logits
    
    def encode(self, src: torch.Tensor, src_pad_idx: int = 0) -> torch.Tensor:
        """
        仅运行 Encoder（用于推理时的编码阶段）
        
        Args:
            src: 源序列 [batch_size, src_len]
            src_pad_idx: 源序列填充 token 索引
        
        Returns:
            encoder_output: [batch_size, src_len, d_model]
        """
        src_mask = self.create_padding_mask(src, src_pad_idx)
        src_embedded = self.src_embedding(src)
        return self.encoder(src_embedded, src_mask)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               src_pad_idx: int = 0, tgt_pad_idx: int = 0) -> torch.Tensor:
        """
        仅运行 Decoder（用于推理时的解码阶段）
        
        Args:
            tgt: 目标序列 [batch_size, tgt_len]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_pad_idx: 源序列填充 token 索引
            tgt_pad_idx: 目标序列填充 token 索引
        
        Returns:
            logits: [batch_size, tgt_len, tgt_vocab_size]
        """
        # 创建 src_mask
        src_seq_len = encoder_output.size(1)
        src_mask = torch.ones(encoder_output.size(0), 1, 1, src_seq_len).to(encoder_output.device)
        
        # 创建 tgt_mask
        tgt_mask = self.create_padding_mask(tgt, tgt_pad_idx)
        tgt_seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(tgt_seq_len).to(tgt.device)
        
        if tgt_mask is not None:
            tgt_mask = tgt_mask & look_ahead_mask.bool()
        else:
            tgt_mask = look_ahead_mask
        
        tgt_embedded = self.tgt_embedding(tgt)
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)
        logits = self.output_projection(decoder_output)
        return logits
