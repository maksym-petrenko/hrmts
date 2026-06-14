"""
Small CNN on CIFAR-10 with WeightWatcher spectral tracking.

Convolutional analogue of mnist_mlp.py: trains a 4-conv + 2-FC CNN and
records per-layer alpha-hat after each epoch to observe whether the
heavy-tailed self-regularization phenomenon also appears in the spectra
of convolutional weight matrices.

Each Conv2d weight tensor of shape (C_out, C_in, k, k) is reshaped into
the 2D matrix of shape (C_out, C_in * k * k) before its WW W^T spectrum
is computed. WeightWatcher applies the same reshape internally.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import weightwatcher as ww

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"


class CNN(nn.Module):
    """4 conv layers (32->64->128->256, 3x3, ReLU, max-pool every other)
    + 2 FC layers (-> 256 -> 10).  ~1.3M parameters."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   32, kernel_size=3, padding=1),     # 32x32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  64, kernel_size=3, padding=1),     # 32x32x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 16x16x64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),     # 16x16x128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),    # 16x16x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                  # 8x8x256
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_dataloaders():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


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


def _weight_matrix(module):
    """Return the 2D matrix associated to a Linear/Conv2d weight tensor."""
    W = module.weight.detach().cpu().numpy().astype(np.float64)
    if W.ndim == 4:
        # (C_out, C_in, k_h, k_w) -> (C_out, C_in * k_h * k_w)
        W = W.reshape(W.shape[0], -1)
    return W


def compute_eigenvalues(model):
    """Compute the eigenvalues of W W^T for every Linear/Conv2d layer."""
    eigs = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            W = _weight_matrix(module)
            M = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
            lam = np.linalg.eigvalsh(M)
            lam = lam[lam > 1e-12]
            eigs[name] = lam.tolist()
    return eigs


def analyze_weights(model, epoch):
    """Run WeightWatcher on current model and return per-layer metrics."""
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(
        min_evals=10,
        plot=False,
    )
    layer_metrics = []
    for _, row in details.iterrows():
        layer_metrics.append({
            "layer_id": int(row["layer_id"]),
            "name": str(row.get("name", "")),
            "alpha": float(row["alpha"]) if "alpha" in row and row["alpha"] == row["alpha"] else None,
            "alpha_weighted": float(row["alpha_weighted"]) if "alpha_weighted" in row and row["alpha_weighted"] == row["alpha_weighted"] else None,
            "log_norm": float(row["log_norm"]) if "log_norm" in row and row["log_norm"] == row["log_norm"] else None,
            "log_spectral_norm": float(row["log_spectral_norm"]) if "log_spectral_norm" in row and row["log_spectral_norm"] == row["log_spectral_norm"] else None,
            "xmin": float(row["xmin"]) if "xmin" in row and row["xmin"] == row["xmin"] else None,
        })
    return layer_metrics


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = CNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model on {DEVICE} with {n_params/1e6:.2f}M parameters")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = get_dataloaders()

    history = []
    eig_history = {}

    # Analyze before training (epoch 0)
    print("Epoch 0 (untrained)")
    acc = evaluate(model, test_loader)
    layer_metrics = analyze_weights(model, 0)
    eig_history[0] = compute_eigenvalues(model)
    record = {"epoch": 0, "test_acc": acc, "layers": layer_metrics}
    history.append(record)
    print(f"  test_acc={acc:.4f}")
    for lm in layer_metrics:
        print(f"  layer {lm['layer_id']} ({lm['name']}): alpha={lm['alpha']}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        acc = evaluate(model, test_loader)
        layer_metrics = analyze_weights(model, epoch)
        # Save eigenvalues only at epoch 0 and the final epoch (saves time/space)
        if epoch == EPOCHS:
            eig_history[epoch] = compute_eigenvalues(model)

        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "test_acc": acc,
            "layers": layer_metrics,
        }
        history.append(record)

        print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={acc:.4f}")
        for lm in layer_metrics:
            print(f"  layer {lm['layer_id']} ({lm['name']}): alpha={lm['alpha']}")

    # Save results
    results_path = RESULTS_DIR / "cnn_tracking.json"
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nMetrics saved to {results_path}")

    eig_path = RESULTS_DIR / "cnn_eigenvalues.json"
    with open(eig_path, "w") as f:
        json.dump({str(k): v for k, v in eig_history.items()}, f)
    print(f"Eigenvalues saved to {eig_path}")

    # Print summary
    print("\n--- Alpha evolution summary ---")
    print(f"{'Epoch':>5}", end="")
    layer_ids = [lm["layer_id"] for lm in history[0]["layers"]]
    for lid in layer_ids:
        print(f"  Layer {lid:>2}", end="")
    print("   Acc")
    for rec in history:
        print(f"{rec['epoch']:>5}", end="")
        for lm in rec["layers"]:
            a = lm["alpha"]
            print(f"  {a:>8.3f}" if a is not None else "      N/A", end="")
        print(f"  {rec['test_acc']:.4f}")


if __name__ == "__main__":
    main()
