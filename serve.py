"""
OpenAI 兼容的推理服务
接口: POST /v1/chat/completions
      GET  /v1/models
"""
import time
import uuid
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformer import Transformer

# ── 模型超参数（必须与 checkpoint 一致） ──────────────────────────
SRC_VOCAB = 1000
TGT_VOCAB = 1000
D_MODEL   = 128
N_HEADS   = 4
D_FF      = 256
N_LAYERS  = 2

SOS, EOS, PAD = 1, 2, 0
MAX_SRC_LEN   = 50
MAX_GEN_LEN   = 50

# ── 加载模型 ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    src_vocab_size=SRC_VOCAB,
    tgt_vocab_size=TGT_VOCAB,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    n_layers=N_LAYERS,
    dropout=0.0,
).to(device)

ckpt = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── 简易 tokenizer（字符级，无真实词表） ─────────────────────────
def tokenize(text: str) -> List[int]:
    """将文本映射为 token id 列表（范围 3-999，避开 PAD/SOS/EOS）"""
    return [ord(c) % 997 + 3 for c in text[:MAX_SRC_LEN]] or [SOS]

def detokenize(ids: List[int]) -> str:
    """将 token id 列表转回字符串表示（玩具模型无真实词表）"""
    return " ".join(str(i) for i in ids)

# ── 贪心解码 ──────────────────────────────────────────────────────
def generate(src_ids: List[int], max_len: int) -> List[int]:
    src = torch.tensor([src_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        enc_out = model.encode(src)
        tgt = torch.tensor([[SOS]], dtype=torch.long).to(device)
        for _ in range(max_len):
            logits   = model.decode(tgt, enc_out)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt      = torch.cat([tgt, next_tok], dim=1)
            if next_tok.item() == EOS:
                break
    return tgt[0, 1:].tolist()  # 去掉 SOS

# ── FastAPI ───────────────────────────────────────────────────────
app = FastAPI(title="Toy Transformer API")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "toy-transformer"
    messages: List[Message]
    max_tokens: Optional[int] = MAX_GEN_LEN

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # 取最后一条 user 消息作为输入
    user_text = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )
    src_ids  = tokenize(user_text)
    out_ids  = generate(src_ids, req.max_tokens or MAX_GEN_LEN)
    content  = detokenize(out_ids)

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   req.model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     len(src_ids),
            "completion_tokens": len(out_ids),
            "total_tokens":      len(src_ids) + len(out_ids),
        },
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id":       "toy-transformer",
            "object":   "model",
            "created":  0,
            "owned_by": "local",
        }],
    }
