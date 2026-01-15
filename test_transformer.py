"""
测试 Transformer 模型
演示模型的各个组件和完整流程
"""
import torch
import torch.nn as nn
from transformer import Transformer
from embeddings import TransformerEmbedding, PositionalEncoding
from attention import MultiHeadAttention, ScaledDotProductAttention
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from modules import PositionwiseFeedForward, ResidualConnection


def test_positional_encoding():
    """测试位置编码"""
    print("=" * 60)
    print("测试位置编码")
    print("=" * 60)
    
    d_model = 512
    max_len = 100
    pos_encoding = PositionalEncoding(d_model, max_len, dropout=0)
    
    # 创建一个 batch 的输入
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = pos_encoding(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"位置编码形状: {pos_encoding.pe.shape}")
    print("✓ 位置编码测试通过\n")


def test_multihead_attention():
    """测试多头注意力"""
    print("=" * 60)
    print("测试多头注意力")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 10
    d_model = 512
    n_heads = 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    
    # 创建 Q, K, V
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    output, attention_weights = mha(query, key, value)
    
    print(f"输入形状: {query.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"每个头的维度: {d_model // n_heads}")
    print("✓ 多头注意力测试通过\n")


def test_feedforward():
    """测试前馈网络"""
    print("=" * 60)
    print("测试前馈网络")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 10
    d_model = 512
    d_ff = 2048
    
    ff = PositionwiseFeedForward(d_model, d_ff, dropout=0)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = ff(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"隐藏层维度: {d_ff}")
    print("✓ 前馈网络测试通过\n")


def test_encoder_layer():
    """测试编码器层"""
    print("=" * 60)
    print("测试编码器层")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 10
    d_model = 512
    n_heads = 8
    d_ff = 2048
    
    encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout=0)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = encoder_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✓ 编码器层测试通过\n")


def test_decoder_layer():
    """测试解码器层"""
    print("=" * 60)
    print("测试解码器层")
    print("=" * 60)
    
    batch_size = 4
    src_len = 10
    tgt_len = 8
    d_model = 512
    n_heads = 8
    d_ff = 2048
    
    decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout=0)
    
    x = torch.randn(batch_size, tgt_len, d_model)
    encoder_output = torch.randn(batch_size, src_len, d_model)
    
    output = decoder_layer(x, encoder_output)
    
    print(f"解码器输入形状: {x.shape}")
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"解码器输出形状: {output.shape}")
    print("✓ 解码器层测试通过\n")


def test_full_transformer():
    """测试完整 Transformer"""
    print("=" * 60)
    print("测试完整 Transformer 模型")
    print("=" * 60)
    
    # 模型配置（小模型用于快速测试）
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 128
    n_heads = 4
    d_ff = 256
    n_layers = 2
    batch_size = 4
    src_len = 10
    tgt_len = 8
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=0
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 测试前向传播
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    
    print(f"\n源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    
    output = model(src, tgt)
    
    print(f"输出形状: {output.shape}")
    print(f"期望形状: ({batch_size}, {tgt_len}, {tgt_vocab_size})")
    
    # 验证输出形状
    assert output.shape == (batch_size, tgt_len, tgt_vocab_size), "输出形状不匹配"
    print("✓ 完整 Transformer 测试通过\n")


def test_inference_mode():
    """测试推理模式"""
    print("=" * 60)
    print("测试推理模式")
    print("=" * 60)
    
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 128
    n_heads = 4
    d_ff = 256
    n_layers = 2
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=0
    )
    
    batch_size = 2
    src_len = 10
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    
    # 1. 编码阶段
    encoder_output = model.encode(src)
    print(f"编码器输出形状: {encoder_output.shape}")
    
    # 2. 解码阶段（逐步生成）
    # 假设我们有一个起始 token
    tgt = torch.tensor([[1], [1]])  # [batch_size, 1]
    
    for i in range(5):
        logits = model.decode(tgt, encoder_output)
        print(f"  步骤 {i+1}: tgt 形状={tgt.shape}, logits 形状={logits.shape}")
        
        # 获取下一个 token（贪婪解码）
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)
    
    print(f"最终生成的序列形状: {tgt.shape}")
    print("✓ 推理模式测试通过\n")


def print_model_summary():
    """打印模型结构摘要"""
    print("=" * 60)
    print("模型结构摘要")
    print("=" * 60)
    
    d_model = 128
    n_heads = 4
    d_ff = 256
    n_layers = 2
    
    print(f"""
配置参数:
- d_model (模型维度): {d_model}
- n_heads (注意力头数): {n_heads}
- d_k (每个头维度): {d_model // n_heads}
- d_ff (前馈隐藏层): {d_ff}
- n_layers (层数): {n_layers}

数据流:
1. 输入 [batch_size, seq_len]
2. Token Embedding -> [batch_size, seq_len, d_model]
3. Positional Encoding -> [batch_size, seq_len, d_model]
4. Encoder (N 层):
   - Multi-Head Self-Attention
   - Add & Norm (残差连接 + 层归一化)
   - Feed-Forward Network
   - Add & Norm
5. Decoder (N 层):
   - Masked Multi-Head Self-Attention
   - Add & Norm
   - Encoder-Decoder Attention
   - Add & Norm
   - Feed-Forward Network
   - Add & Norm
6. Output Projection -> [batch_size, seq_len, vocab_size]
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Transformer 模型测试")
    print("=" * 60 + "\n")
    
    # 打印模型摘要
    print_model_summary()
    
    # 依次测试各个组件
    test_positional_encoding()
    test_multihead_attention()
    test_feedforward()
    test_encoder_layer()
    test_decoder_layer()
    test_full_transformer()
    test_inference_mode()
    
    print("=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
