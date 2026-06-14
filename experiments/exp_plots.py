import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

R = Path(__file__).parent / "results"
F = Path(__file__).parent.parent / "figures"
F.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 120})


def plot_mnist_alpha():
    log = json.loads((R / "mnist_alpha.json").read_text())
    aps = log["alpha"]
    keys = sorted(int(k) for k in aps[0].keys())
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for i, k in enumerate(keys):
        ax.plot(range(len(aps)), [a[str(k)] for a in aps],
                marker="o", markersize=2.5, linewidth=1.2, label=f"layer {i+1}")
    ax.axhline(2, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(4, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\alpha$")
    ax.set_yscale("log")
    ax.set_yticks([2, 3, 5, 10, 20, 50])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(F / "exp_mnist_alpha.pdf")
    fig.savefig(F / "exp_mnist_alpha.png")
    plt.close(fig)


def plot_mnist_esd():
    init = json.loads((R / "mnist_evs_init.json").read_text())
    final = json.loads((R / "mnist_evs_final.json").read_text())
    keys = list(init.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(2.4 * len(keys), 2.8))
    if len(keys) == 1:
        axes = [axes]
    for i, (ax, k) in enumerate(zip(axes, keys)):
        ei = np.array(init[k]); ef = np.array(final[k])
        ei = ei[ei > 1e-14]; ef = ef[ef > 1e-14]
        lo = min(ei.min(), ef.min()); hi = max(ei.max(), ef.max())
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
        ax.hist(ei, bins=bins, alpha=0.5, density=True, label="init")
        ax.hist(ef, bins=bins, alpha=0.5, density=True, label="trained")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"layer {i+1}", fontsize=9)
        ax.set_xlabel(r"$\lambda$")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(F / "exp_mnist_esd.pdf")
    fig.savefig(F / "exp_mnist_esd.png")
    plt.close(fig)


def plot_resnet():
    d = json.loads((R / "resnet_alpha.json").read_text())["alphas"]
    items = sorted(d.items(), key=lambda kv: kv[1]["layer_id"])
    names = [v["longname"] for _, v in items]
    alphas = [v["alpha"] for _, v in items]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.bar(range(len(names)), alphas, color="steelblue")
    ax.axhline(2, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(4, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel(r"$\alpha$")
    fig.tight_layout()
    fig.savefig(F / "exp_resnet_alpha.pdf")
    fig.savefig(F / "exp_resnet_alpha.png")
    plt.close(fig)


def plot_resnet_esd():
    d = json.loads((R / "resnet_alpha.json").read_text())
    evs = d["evs"]
    alphas = d["alphas"]
    name_to_alpha = {}
    for _, v in alphas.items():
        ln = v["longname"]
        name_to_alpha[ln] = v["alpha"]
    wanted = ["layer1.0.conv1.weight", "layer2.1.conv1.weight",
              "layer3.0.conv2.weight", "fc.weight"]
    picks = [n for n in wanted if n in evs]
    fig, axes = plt.subplots(1, len(picks), figsize=(2.4 * len(picks), 2.8))
    if len(picks) == 1:
        axes = [axes]
    for ax, n in zip(axes, picks):
        ev = np.array(evs[n])
        ev = ev[ev > 0]
        bins = np.logspace(np.log10(ev.min()), np.log10(ev.max()), 50)
        ax.hist(ev, bins=bins, density=True, color="steelblue")
        ax.set_xscale("log")
        ax.set_yscale("log")
        key = n.replace(".weight", "")
        a = name_to_alpha.get(key, float("nan"))
        ax.set_title(f"{key}\n" + r"$\alpha=$" + f"{a:.2f}", fontsize=7)
        ax.set_xlabel(r"$\lambda$")
    fig.tight_layout()
    fig.savefig(F / "exp_resnet_esd.pdf")
    fig.savefig(F / "exp_resnet_esd.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_mnist_alpha()
    plot_mnist_esd()
    plot_resnet()
    plot_resnet_esd()
