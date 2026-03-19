# transformer-practice

从零手写 Transformer，并提供 OpenAI 兼容的推理接口。

## 项目结构

```
├── attention.py       # 多头注意力
├── embeddings.py      # Token 嵌入 + 位置编码
├── encoder.py         # Encoder
├── decoder.py         # Decoder
├── modules.py         # FFN、残差连接
├── transformer.py     # 完整模型
├── simple_train.py    # 训练示例（随机数据）
├── serve.py           # FastAPI 推理服务
├── checkpoint.pth     # 已训练权重
├── pyproject.toml     # 依赖配置（uv）
└── Dockerfile
```

## 模型参数

| 参数 | 值 |
|------|----|
| vocab_size | 1000 |
| d_model | 128 |
| n_heads | 4 |
| d_ff | 256 |
| n_layers | 2 |

## 快速开始

**安装依赖（需要 [uv](https://github.com/astral-sh/uv)）**

```bash
uv sync
```

**训练**

```bash
uv run python simple_train.py
```

**启动推理服务**

```bash
uv run uvicorn serve:app --host 0.0.0.0 --port 8000
```

## API 使用

接口兼容 OpenAI Chat Completions 格式。

**PowerShell**
```powershell
Invoke-WebRequest -Uri http://localhost:8000/v1/chat/completions `
  -Method Post `
  -Headers @{"Content-Type"="application/json"} `
  -Body '{"model":"toy-transformer","messages":[{"role":"user","content":"hello"}]}'
```

**curl（Linux/macOS）**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"toy-transformer","messages":[{"role":"user","content":"hello"}]}'
```

**curl（Windows CMD）**
```cmd
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"toy-transformer\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
```

**预期响应**
```json
{
  "id":"chatcmpl-a5e0697e",        // 本次请求的唯一标识
  "object":"chat.completion",      // 返回对象类型（符合 OpenAI API 规范）
  "created":1773889266,            // 请求创建时间戳
  "model":"toy-transformer",       // 使用的模型（和你请求的一致）
  "choices":[{                     // 回复结果列表
    "index":0,
    "message":{
      "role":"assistant",          // 角色（助手回复）
      "content":"621 621 ... 119"  // 模型生成的回复内容
    },
    "finish_reason":"stop"         // 结束原因（stop 表示正常结束）
  }],
  "usage":{                        // 令牌使用统计
    "prompt_tokens":5,             // 提问消耗的令牌数
    "completion_tokens":50,        // 回复消耗的令牌数
    "total_tokens":55              // 总令牌数
  }
}
```

> 注意：模型使用随机数据训练，输出为 token ID 序列，无实际语义。

## Docker 部署

```bash
docker build -t toy-transformer .
docker run -p 8000:8000 toy-transformer
```

**查看本地镜像**

```bash
docker images toy-transformer
```

**导出为文件（离线传输）**

```bash
# 导出
docker save -o toy-transformer.tar toy-transformer

# 在另一台机器上导入
docker load -i toy-transformer.tar
```

**推送到镜像仓库（推荐用于云端部署）**

```bash
docker tag toy-transformer your-username/toy-transformer
docker push your-username/toy-transformer
```
