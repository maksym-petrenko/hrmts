"""
Tiny character-level GPT on Tiny Shakespeare with WeightWatcher tracking.

Trains a small decoder-only transformer from scratch and records per-projection
alpha-hat (Q, K, V, O, FFN1, FFN2 in every block, plus token/pos embeddings
and LM head) every TRACK_EVERY steps -- analogue of the MNIST MLP and ViT
experiments, but indexed by training step instead of epoch.
"""

import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import weightwatcher as ww

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training
TOTAL_STEPS = 4000
BATCH_SIZE = 64
BLOCK_SIZE = 128
LR = 3e-4
WD = 0.0
WARMUP = 200
TRACK_EVERY = 500   # WW analyze cadence
EVAL_EVERY = 500
EVAL_BATCHES = 20

# Architecture (~2M params with these settings)
EMBED = 192
DEPTH = 6
HEADS = 6
DROPOUT = 0.0
MLP_RATIO = 4

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
DATA_PATH = Path(__file__).parent / "data" / "tinyshakespeare.txt"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_text():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def encode_split():
    text = get_text()
    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    train = torch.from_numpy(data[:n])
    val = torch.from_numpy(data[n:])
    return train, val, len(chars)


def get_batch(data, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CausalAttention(nn.Module):
    def __init__(self, embed=EMBED, heads=HEADS, block_size=BLOCK_SIZE):
        super().__init__()
        assert embed % heads == 0
        self.heads = heads
        self.head_dim = embed // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(embed, embed, bias=False)
        self.k = nn.Linear(embed, embed, bias=False)
        self.v = nn.Linear(embed, embed, bias=False)
        self.o = nn.Linear(embed, embed)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, N, D = x.shape
        H, d = self.heads, self.head_dim
        q = self.q(x).view(B, N, H, d).transpose(1, 2)
        k = self.k(x).view(B, N, H, d).transpose(1, 2)
        v = self.v(x).view(B, N, H, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :N, :N] == 0, float("-inf"))
        att = att.softmax(dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, embed=EMBED, ratio=MLP_RATIO):
        super().__init__()
        hidden = int(embed * ratio)
        self.fc1 = nn.Linear(embed, hidden)
        self.fc2 = nn.Linear(hidden, embed)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, embed=EMBED, heads=HEADS, block_size=BLOCK_SIZE):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed)
        self.attn = CausalAttention(embed, heads, block_size)
        self.norm2 = nn.LayerNorm(embed)
        self.mlp = MLP(embed)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed=EMBED, depth=DEPTH, heads=HEADS,
                 block_size=BLOCK_SIZE):
        super().__init__()
        self.block_size = block_size
        self.tok_embed = nn.Embedding(vocab_size, embed)
        self.pos_embed = nn.Embedding(block_size, embed)
        self.blocks = nn.ModuleList(
            [Block(embed, heads, block_size) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, idx, targets=None):
        B, N = idx.shape
        pos = torch.arange(0, N, device=idx.device)
        x = self.tok_embed(idx) + self.pos_embed(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Eval / WW
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train, val):
    model.eval()
    out = {}
    for name, data in (("train", train), ("val", val)):
        losses = []
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    model.train()
    return out


def compute_eigenvalues(model):
    eigs = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy().astype(np.float64)
            M = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
            lam = np.linalg.eigvalsh(M)
            lam = lam[lam > 1e-12]
            eigs[name] = lam.tolist()
    return eigs


def _safe(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def analyze_weights(model):
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(min_evals=10, plot=False)
    layer_metrics = []
    for _, row in details.iterrows():
        layer_metrics.append({
            "layer_id": int(row["layer_id"]),
            "name": str(row.get("name", "")),
            "longname": str(row.get("longname", "")),
            "alpha": _safe(row.get("alpha")),
            "alpha_weighted": _safe(row.get("alpha_weighted")),
            "log_norm": _safe(row.get("log_norm")),
            "log_spectral_norm": _safe(row.get("log_spectral_norm")),
            "xmin": _safe(row.get("xmin")),
        })
    return layer_metrics


def cosine_lr(step, total, base, warm):
    if step < warm:
        return base * step / max(1, warm)
    progress = (step - warm) / max(1, total - warm)
    return 0.5 * base * (1.0 + math.cos(math.pi * progress))


def _kind(longname: str) -> str:
    n = longname
    if n.endswith(".q"):
        return "q"
    if n.endswith(".k"):
        return "k"
    if n.endswith(".v"):
        return "v"
    if n.endswith(".o"):
        return "o"
    if n.endswith(".fc1"):
        return "fc1"
    if n.endswith(".fc2"):
        return "fc2"
    if n == "head":
        return "head"
    return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train, val, vocab = encode_split()
    print(f"vocab={vocab}  train={len(train)}  val={len(val)}  device={DEVICE}")

    model = TinyGPT(vocab).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyGPT params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD,
                                  betas=(0.9, 0.95))

    history = []
    eig_history = {}

    # Initial snapshot
    losses = estimate_loss(model, train, val)
    layer_metrics = analyze_weights(model)
    eig_history[0] = compute_eigenvalues(model)
    history.append({
        "step": 0,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "test_acc": None,
        "layers": layer_metrics,
    })
    print(f"Step 0  train={losses['train']:.4f}  val={losses['val']:.4f}  "
          f"layers tracked={len(layer_metrics)}")

    t0 = time.time()
    running = 0.0
    n_running = 0
    for step in range(1, TOTAL_STEPS + 1):
        lr_now = cosine_lr(step, TOTAL_STEPS, LR, WARMUP)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        x, y = get_batch(train)
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item()
        n_running += 1

        if step % TRACK_EVERY == 0 or step == TOTAL_STEPS:
            losses = estimate_loss(model, train, val)
            layer_metrics = analyze_weights(model)

            avg_loss = running / max(1, n_running)
            running = 0.0
            n_running = 0

            history.append({
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "running_loss": avg_loss,
                "test_acc": None,
                "layers": layer_metrics,
            })
            elapsed = time.time() - t0
            print(f"Step {step}/{TOTAL_STEPS}  train={losses['train']:.4f}  "
                  f"val={losses['val']:.4f}  lr={lr_now:.2e}  "
                  f"elapsed={elapsed:.0f}s")
            by_kind = {}
            for lm in layer_metrics:
                k = _kind(lm["longname"])
                if lm["alpha"] is not None and 0 < lm["alpha"] < 20:
                    by_kind.setdefault(k, []).append(lm["alpha"])
            for k in ("q", "k", "v", "o", "fc1", "fc2", "head"):
                if k in by_kind:
                    vs = by_kind[k]
                    print(f"   {k:>5s}  mean alpha={np.mean(vs):.3f}  n={len(vs)}")

    eig_history[TOTAL_STEPS] = compute_eigenvalues(model)

    out_track = RESULTS_DIR / "gpt_tracking.json"
    with open(out_track, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nMetrics saved to {out_track}")

    out_eigs = RESULTS_DIR / "gpt_eigenvalues.json"
    with open(out_eigs, "w") as f:
        json.dump({str(k): v for k, v in eig_history.items()}, f)
    print(f"Eigenvalues saved to {out_eigs}")


if __name__ == "__main__":
    main()
