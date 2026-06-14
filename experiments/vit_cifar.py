"""
Tiny Vision Transformer on CIFAR-10 with WeightWatcher spectral tracking.

Trains a small ViT from scratch and records per-projection alpha-hat
(Q, K, V, O, FFN1, FFN2 in every transformer block, plus patch embed and head)
after each epoch, to observe the heavy-tailed self-regularization phenomenon
in attention-based architectures (analogue of mnist_mlp.py).
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import weightwatcher as ww

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 128
LR = 3e-4
WD = 1e-4

# ViT hyperparameters
IMG_SIZE = 32
PATCH = 4
EMBED = 192
DEPTH = 6
HEADS = 6
MLP_RATIO = 4
NUM_CLASSES = 10
DROPOUT = 0.0

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Linear (Conv2d-as-linear) patch embedding. Single Linear-like weight."""

    def __init__(self, img_size=IMG_SIZE, patch=PATCH, in_chans=3, embed=EMBED):
        super().__init__()
        assert img_size % patch == 0
        self.n_patches = (img_size // patch) ** 2
        # Use an explicit nn.Linear so WeightWatcher picks it up cleanly.
        self.patch = patch
        self.proj = nn.Linear(in_chans * patch * patch, embed)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch
        # (B, C, H, W) -> (B, n_patches, C*p*p)
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B,C,H/p,W/p,p,p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, self.n_patches, C * p * p)
        return self.proj(x)


class Attention(nn.Module):
    """Self-attention with separate q, k, v, o linears (no bias on QKV)."""

    def __init__(self, embed=EMBED, heads=HEADS):
        super().__init__()
        assert embed % heads == 0
        self.heads = heads
        self.head_dim = embed // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(embed, embed, bias=False)
        self.k = nn.Linear(embed, embed, bias=False)
        self.v = nn.Linear(embed, embed, bias=False)
        self.o = nn.Linear(embed, embed)

    def forward(self, x):
        B, N, D = x.shape
        H = self.heads
        d = self.head_dim
        q = self.q(x).view(B, N, H, d).transpose(1, 2)
        k = self.k(x).view(B, N, H, d).transpose(1, 2)
        v = self.v(x).view(B, N, H, d).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
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
    def __init__(self, embed=EMBED, heads=HEADS):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed)
        self.attn = Attention(embed, heads)
        self.norm2 = nn.LayerNorm(embed)
        self.mlp = MLP(embed)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=IMG_SIZE,
        patch=PATCH,
        embed=EMBED,
        depth=DEPTH,
        heads=HEADS,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        self.embed = embed
        self.patch_embed = PatchEmbed(img_size, patch, 3, embed)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed))
        self.blocks = nn.ModuleList([Block(embed, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_dataloaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
    pin = DEVICE.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=pin
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=pin
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Eval / metric extraction
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def compute_eigenvalues(model):
    """Compute eigenvalues of W W^T for every Linear, keyed by dotted name."""
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
    """Return float or None for NaNs / missing fields."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def analyze_weights(model):
    """Run WeightWatcher and return per-layer metrics."""
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


def cosine_lr(step, total, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = ViT().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT params: {n_params/1e6:.2f}M  device={DEVICE}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_dataloaders()
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = len(train_loader)  # 1 epoch warmup

    history = []
    eig_history = {}

    # Epoch 0 (init)
    print("Epoch 0 (untrained)")
    acc0 = evaluate(model, test_loader)
    layer_metrics = analyze_weights(model)
    eig_history[0] = compute_eigenvalues(model)
    history.append({
        "epoch": 0,
        "test_acc": acc0,
        "train_loss": None,
        "layers": layer_metrics,
    })
    print(f"  test_acc={acc0:.4f}  layers tracked={len(layer_metrics)}")

    step = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            lr_now = cosine_lr(step, total_steps, LR, warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            step += 1
        avg_loss = total_loss / max(1, n_batches)

        acc = evaluate(model, test_loader)
        layer_metrics = analyze_weights(model)

        # Save eigenvalues only at init and final to keep file small.
        if epoch == EPOCHS:
            eig_history[epoch] = compute_eigenvalues(model)

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "test_acc": acc,
            "layers": layer_metrics,
        })

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={acc:.4f}  "
              f"lr={lr_now:.2e}  elapsed={elapsed:.0f}s")

        # Print a compact alpha summary by projection type.
        by_kind = {}
        for lm in layer_metrics:
            kind = _kind(lm["longname"])
            if lm["alpha"] is not None and 0 < lm["alpha"] < 20:
                by_kind.setdefault(kind, []).append(lm["alpha"])
        for kind in ("q", "k", "v", "o", "fc1", "fc2", "patch", "head"):
            if kind in by_kind:
                vals = by_kind[kind]
                print(f"   {kind:>5s}  mean alpha={np.mean(vals):.3f}  n={len(vals)}")

    # Save tracking JSON
    out_track = RESULTS_DIR / "vit_tracking.json"
    with open(out_track, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nMetrics saved to {out_track}")

    out_eigs = RESULTS_DIR / "vit_eigenvalues.json"
    with open(out_eigs, "w") as f:
        json.dump({str(k): v for k, v in eig_history.items()}, f)
    print(f"Eigenvalues saved to {out_eigs}")


def _kind(longname: str) -> str:
    """Classify a Linear layer by projection type (q/k/v/o/fc1/fc2/patch/head)."""
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
    if "patch_embed" in n:
        return "patch"
    if n == "head":
        return "head"
    return "other"


if __name__ == "__main__":
    main()
