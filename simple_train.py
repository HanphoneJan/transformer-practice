"""
简单的训练示例（演示用）
使用随机数据演示如何训练 Transformer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer


def generate_dummy_data(batch_size, src_len, tgt_len, src_vocab_size, tgt_vocab_size, device):
    """生成随机训练数据"""
    src = torch.randint(0, src_vocab_size, (batch_size, src_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len)).to(device)
    return src, tgt


def train_simple():
    """简单训练演示"""
    print("=" * 60)
    print("简单训练演示")
    print("=" * 60 + "\n")
    
    # 超参数配置（小模型用于快速演示）
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 128
    n_heads = 4
    d_ff = 256
    n_layers = 2
    dropout = 0.1
    
    batch_size = 32
    src_len = 20
    tgt_len = 15
    
    learning_rate = 0.0001
    n_epochs = 10
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}\n")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding token
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # 训练循环
    print("开始训练...\n")
    model.train()
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 10  # 每个 epoch 训练 10 个 batch
        
        for batch_idx in range(n_batches):
            # 生成随机数据
            src, tgt = generate_dummy_data(
                batch_size, src_len, tgt_len,
                src_vocab_size, tgt_vocab_size, device
            )
            
            # 前向传播
            # src: [batch_size, src_len]
            # tgt: [batch_size, tgt_len]
            logits = model(src, tgt)
            # logits: [batch_size, tgt_len, tgt_vocab_size]
            
            # 计算损失（预测除第一个 token 外的所有 token）
            # 移除最后一个输出的 logits（因为最后没有对应的目标）
            # 移除第一个目标 token（用于预测的输入）
            logits = logits[:, :-1, :].contiguous()
            logits = logits.view(-1, tgt_vocab_size)
            
            tgt_out = tgt[:, 1:].contiguous()
            tgt_out = tgt_out.view(-1)
            
            loss = criterion(logits, tgt_out)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        # 更新学习率
        scheduler.step()
        
        avg_loss = total_loss / n_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{n_epochs}] | "
              f"Loss: {avg_loss:.4f} | "
              f"LR: {current_lr:.6f}")
    
    print("\n训练完成！")
    print("\n" + "=" * 60)
    print("保存模型")
    print("=" * 60)
    
    # 保存模型
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, 'checkpoint.pth')
    
    print("模型已保存到 checkpoint.pth")


def inference_demo():
    """推理演示"""
    print("\n" + "=" * 60)
    print("推理演示")
    print("=" * 60 + "\n")
    
    # 配置
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 128
    n_heads = 4
    d_ff = 256
    n_layers = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers
    ).to(device)
    
    model.eval()
    
    # 生成源序列
    batch_size = 1
    src_len = 10
    src = torch.randint(0, src_vocab_size, (batch_size, src_len)).to(device)
    
    print(f"源序列: {src.tolist()}")
    
    # 编码
    with torch.no_grad():
        encoder_output = model.encode(src)
        print(f"编码器输出形状: {encoder_output.shape}")
        
        # 逐步解码
        max_len = 15
        sos_token = 1  # 起始 token
        eos_token = 2  # 结束 token
        
        tgt = torch.tensor([[sos_token]]).to(device)
        print(f"\n起始 token: {tgt.item()}")
        
        for i in range(max_len):
            logits = model.decode(tgt, encoder_output)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            print(f"步骤 {i+2}: 生成的 token: {next_token.item()}")
            
            # 如果生成了结束 token，停止生成
            if next_token.item() == eos_token:
                break
        
        print(f"\n生成的完整序列: {tgt.tolist()}")
        print(f"序列长度: {tgt.size(1)}")


if __name__ == "__main__":
    # 训练
    train_simple()
    
    # 推理
    inference_demo()
