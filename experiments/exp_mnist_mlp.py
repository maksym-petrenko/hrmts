import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import weightwatcher as ww

D = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 100, 64, 0.05
TRAIN_N = 2000
WIDTHS = [784, 2048, 2048, 2048, 2048, 10]
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        L = [nn.Flatten()]
        for i in range(len(WIDTHS) - 1):
            L.append(nn.Linear(WIDTHS[i], WIDTHS[i + 1]))
            if i < len(WIDTHS) - 2:
                L.append(nn.ReLU())
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


def ww_alpha(model):
    cpu = MLP()
    cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    d = ww.WeightWatcher(model=cpu).analyze(plot=False, randomize=False, mp_fit=False)
    return {int(r["layer_id"]): float(r["alpha"]) for _, r in d.iterrows()}


def eigs(model):
    out = {}
    for n, p in model.named_parameters():
        if p.ndim == 2 and "weight" in n:
            sv = torch.linalg.svdvals(p.detach().float())
            out[n] = (sv ** 2).cpu().numpy().tolist()
    return out


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    full = datasets.MNIST("./data", True, download=True, transform=tf)
    sub = torch.utils.data.Subset(full, list(range(TRAIN_N)))
    tr = torch.utils.data.DataLoader(sub, BS, shuffle=True)
    te = torch.utils.data.DataLoader(datasets.MNIST("./data", False, transform=tf), BS)

    model = MLP().to(D)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    init_evs = eigs(model)
    log = {"alpha": [ww_alpha(model)], "acc": []}

    for ep in range(EPOCHS):
        model.train()
        for x, y in tr:
            x, y = x.to(D), y.to(D)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        model.eval()
        c = n = 0
        with torch.no_grad():
            for x, y in te:
                x, y = x.to(D), y.to(D)
                c += (model(x).argmax(1) == y).sum().item()
                n += y.numel()
        log["alpha"].append(ww_alpha(model))
        log["acc"].append(c / n)
        print(f"ep {ep+1:03d} acc {c/n:.4f}")

    final_evs = eigs(model)
    (OUT / "mnist_alpha.json").write_text(json.dumps(log))
    (OUT / "mnist_evs_init.json").write_text(json.dumps(init_evs))
    (OUT / "mnist_evs_final.json").write_text(json.dumps(final_evs))


if __name__ == "__main__":
    main()
