"""
Simple MLP on MNIST with WeightWatcher spectral tracking.

Trains a 3-layer MLP and records per-layer alpha-hat after each epoch
to observe the heavy-tailed self-regularization phenomenon.
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
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
WIDTHS = [784, 2048, 1024, 512, 256, 10]
RESULTS_DIR = Path(__file__).parent / "results"


class MLP(nn.Module):
    def __init__(self, widths=WIDTHS):
        super().__init__()
        layers = [nn.Flatten()]
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_dataloaders():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("./data", train=False, transform=tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1024, shuffle=False
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


def compute_eigenvalues(model):
    """Compute the eigenvalues of W W^T for every Linear layer."""
    eigs = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            W = module.weight.detach().cpu().numpy().astype(np.float64)
            # Use the smaller side for efficiency
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = MLP().to(DEVICE)
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
        print(f"  layer {lm['layer_id']}: alpha={lm['alpha']}")

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
            print(f"  layer {lm['layer_id']}: alpha={lm['alpha']}")

    # Save results
    results_path = RESULTS_DIR / "ww_tracking.json"
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nMetrics saved to {results_path}")

    eig_path = RESULTS_DIR / "eigenvalues.json"
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
