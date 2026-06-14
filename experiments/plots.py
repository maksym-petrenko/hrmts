"""
Generate thesis figures from the training-experiment outputs.

Produces (MLP):
  - fig_alpha_evolution.pdf      : per-layer alpha(t) and test accuracy(t)
  - fig_esd_comparison.pdf       : log-log ESD of each layer at epoch 0 vs final

Produces (CNN, if cnn_tracking.json / cnn_eigenvalues.json exist):
  - fig_cnn_alpha_evolution.pdf  : per-layer alpha(t) and test accuracy(t)
  - fig_cnn_esd_comparison.pdf   : log-log ESD of each layer at epoch 0 vs final
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# WeightWatcher layer_id -> (nn.Module name used in eigenvalues.json, label, color)
LAYERS = [
    (3,  "net.1", r"L1: $784 \times 2048$",   "#1f77b4"),
    (5,  "net.3", r"L2: $2048 \times 1024$",  "#d62728"),
    (7,  "net.5", r"L3: $1024 \times 512$",   "#2ca02c"),
    (9,  "net.7", r"L4: $512 \times 256$",    "#9467bd"),
    (11, "net.9", r"L5: $256 \times 10$",     "#8c564b"),
]


def load():
    with open(RESULTS_DIR / "ww_tracking.json") as f:
        history = json.load(f)
    with open(RESULTS_DIR / "eigenvalues.json") as f:
        eigs = {int(k): v for k, v in json.load(f).items()}
    return history, eigs


def plot_alpha_evolution(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)

    epochs = [r["epoch"] for r in history]

    alpha_by_id = {lid: [] for lid, _, _, _ in LAYERS}
    for rec in history:
        for lm in rec["layers"]:
            if lm["layer_id"] in alpha_by_id:
                alpha_by_id[lm["layer_id"]].append(lm["alpha"])

    off_scale_notes = []
    for lid, _, label, color in LAYERS:
        ys = alpha_by_id[lid]
        ax1.plot(epochs, ys, marker="o", color=color, label=label)
        if ys[0] > 5.5:
            off_scale_notes.append((label.split(":")[0], ys[0], color))

    ax1.axhline(2.0, color="gray", linestyle=":", linewidth=1)
    ax1.axhline(4.0, color="gray", linestyle=":", linewidth=1)
    ax1.text(epochs[-1] - 0.1, 2.05, r"$\alpha=2$ (very heavy-tailed)",
             fontsize=8, color="gray", ha="right", va="bottom")
    ax1.text(epochs[-1] - 0.1, 4.05, r"$\alpha=4$ (Marchenko--Pastur bulk)",
             fontsize=8, color="gray", ha="right", va="bottom")

    if off_scale_notes:
        note = "Epoch 0 (init, off-scale):\n" + "\n".join(
            fr"  {lbl}: $\hat\alpha_0={v:.1f}$" for lbl, v, _ in off_scale_notes
        )
        ax1.text(
            0.02, 0.98, note,
            transform=ax1.transAxes, fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )

    ax1.set_ylim(1.3, 5.5)
    ax1.set_ylabel(r"Power-law exponent $\hat\alpha$")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Spectral evolution during MNIST training")

    acc = [r["test_acc"] for r in history]
    ax2.plot(epochs, acc, marker="s", color="black")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "fig_alpha_evolution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_esd_comparison(history, eigs):
    epochs_sorted = sorted(eigs.keys())
    final_epoch = epochs_sorted[-1]

    final_alphas = {
        lm["layer_id"]: (lm["alpha"], lm.get("xmin"))
        for lm in history[-1]["layers"]
    }

    n_layers = len(LAYERS)
    ncols = 3
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.6 * nrows))
    axes_flat = axes.flatten()

    for ax, (lid, mod_name, label, color) in zip(axes_flat, LAYERS):
        lam0 = np.array(eigs[0][mod_name])
        lamT = np.array(eigs[final_epoch][mod_name])

        bins = np.logspace(
            np.log10(min(lam0.min(), lamT.min())),
            np.log10(max(lam0.max(), lamT.max())),
            40,
        )

        ax.hist(lam0, bins=bins, density=True, alpha=0.45,
                color="gray", label="epoch 0 (init)")
        ax.hist(lamT, bins=bins, density=True, alpha=0.55,
                color=color, label=f"epoch {final_epoch}")

        alpha_hat, xmin = final_alphas[lid]
        if alpha_hat is not None and xmin is not None:
            xs = np.logspace(np.log10(xmin), np.log10(lamT.max()), 100)
            C = (alpha_hat - 1) * xmin ** (alpha_hat - 1)
            ys = C * xs ** (-alpha_hat)
            ax.plot(xs, ys, "k--", linewidth=1.3,
                    label=fr"$\hat\alpha={alpha_hat:.2f}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Eigenvalue $\lambda$")
        ax.set_title(label)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower left")

    for idx in range(n_layers, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for row in range(nrows):
        axes[row, 0].set_ylabel(r"Density $\rho(\lambda)$") if nrows > 1 else axes_flat[0].set_ylabel(r"Density $\rho(\lambda)$")

    fig.suptitle(r"Empirical spectral distribution of $W W^{\!\top}$: initialization vs. trained",
                 y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_esd_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------------
# CNN analogue (CIFAR-10)
# ------------------------------------------------------------------

# WeightWatcher layer_id -> (nn.Module name used in cnn_eigenvalues.json,
# label, color).  Colors match the MLP palette where natural; two extras
# added for the two extra layers.
CNN_LAYERS = [
    # convolutional block (C_in*k*k -> C_out, reshaped)
    ("features.0",  r"C1 conv: $3{\times}3{\times}3 \to 32$",    "#1f77b4"),
    ("features.2",  r"C2 conv: $3{\times}3{\times}32 \to 64$",   "#d62728"),
    ("features.5",  r"C3 conv: $3{\times}3{\times}64 \to 128$",  "#2ca02c"),
    ("features.7",  r"C4 conv: $3{\times}3{\times}128 \to 256$", "#9467bd"),
    # fully-connected head
    ("classifier.1", r"FC1: $16384 \to 256$",                    "#ff7f0e"),
    ("classifier.3", r"FC2: $256 \to 10$",                       "#8c564b"),
]


def load_cnn():
    with open(RESULTS_DIR / "cnn_tracking.json") as f:
        history = json.load(f)
    with open(RESULTS_DIR / "cnn_eigenvalues.json") as f:
        eigs = {int(k): v for k, v in json.load(f).items()}
    return history, eigs


def _save(fig, name):
    """Save fig as both .pdf and .png sibling."""
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def _cnn_layer_roster(history):
    """Return a list of (layer_id, mod_name, label, color), using CNN_LAYERS
    names to look up the layer_id reported by WeightWatcher in ``history``."""
    # Map nn.Module name -> layer_id from the first record
    name_to_id = {}
    for lm in history[0]["layers"]:
        name_to_id[lm["name"]] = lm["layer_id"]

    roster = []
    for mod_name, label, color in CNN_LAYERS:
        # WeightWatcher reports the nn.Module short name without the parent
        # prefix (e.g. "0", "2"); try several variants for robustness.
        candidates = [
            mod_name,
            mod_name.split(".")[-1],
            mod_name.replace("features.", ""),
            mod_name.replace("classifier.", ""),
        ]
        lid = None
        for c in candidates:
            if c in name_to_id:
                lid = name_to_id[c]
                break
        if lid is None:
            # WW may rename; skip gracefully
            print(f"[cnn plot] warning: could not resolve {mod_name} in ww tracking")
            continue
        roster.append((lid, mod_name, label, color))
    return roster


def plot_cnn_alpha_evolution(history):
    roster = _cnn_layer_roster(history)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.8), sharex=True)

    epochs = [r["epoch"] for r in history]

    alpha_by_id = {lid: [] for lid, _, _, _ in roster}
    for rec in history:
        for lm in rec["layers"]:
            if lm["layer_id"] in alpha_by_id:
                alpha_by_id[lm["layer_id"]].append(lm["alpha"])

    off_scale_notes = []
    for lid, _, label, color in roster:
        ys = alpha_by_id[lid]
        # ys may contain None for small layers: drop None and align x
        xs_clean = [e for e, y in zip(epochs, ys) if y is not None]
        ys_clean = [y for y in ys if y is not None]
        if not ys_clean:
            continue
        ax1.plot(xs_clean, ys_clean, marker="o", color=color, label=label)
        if ys_clean[0] is not None and ys_clean[0] > 5.5:
            off_scale_notes.append((label.split(":")[0], ys_clean[0], color))

    ax1.axhline(2.0, color="gray", linestyle=":", linewidth=1)
    ax1.axhline(4.0, color="gray", linestyle=":", linewidth=1)
    ax1.text(epochs[-1] - 0.1, 2.05, r"$\alpha=2$ (very heavy-tailed)",
             fontsize=8, color="gray", ha="right", va="bottom")
    ax1.text(epochs[-1] - 0.1, 4.05, r"$\alpha=4$ (Marchenko--Pastur bulk)",
             fontsize=8, color="gray", ha="right", va="bottom")

    if off_scale_notes:
        note = "Epoch 0 (init, off-scale):\n" + "\n".join(
            fr"  {lbl}: $\hat\alpha_0={v:.1f}$" for lbl, v, _ in off_scale_notes
        )
        ax1.text(
            0.02, 0.98, note,
            transform=ax1.transAxes, fontsize=7.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )

    ax1.set_ylim(1.3, 5.5)
    ax1.set_ylabel(r"Power-law exponent $\hat\alpha$")
    ax1.legend(loc="upper right", ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Spectral evolution during CIFAR-10 training (CNN)")

    acc = [r["test_acc"] for r in history]
    ax2.plot(epochs, acc, marker="s", color="black")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig_cnn_alpha_evolution")


def plot_cnn_esd_comparison(history, eigs):
    roster = _cnn_layer_roster(history)
    epochs_sorted = sorted(eigs.keys())
    final_epoch = epochs_sorted[-1]

    final_alphas = {
        lm["layer_id"]: (lm["alpha"], lm.get("xmin"))
        for lm in history[-1]["layers"]
    }

    n_layers = len(roster)
    ncols = 3
    nrows = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.6 * nrows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else np.array([axes]).flatten()

    for ax, (lid, mod_name, label, color) in zip(axes_flat, roster):
        # eigenvalues are keyed by the full nn.Module name ("features.0", ...)
        key = mod_name if mod_name in eigs[0] else mod_name.split(".")[-1]
        if key not in eigs[0] or key not in eigs[final_epoch]:
            ax.set_visible(False)
            continue
        lam0 = np.array(eigs[0][key])
        lamT = np.array(eigs[final_epoch][key])
        if lam0.size == 0 or lamT.size == 0:
            ax.set_visible(False)
            continue

        bins = np.logspace(
            np.log10(min(lam0.min(), lamT.min())),
            np.log10(max(lam0.max(), lamT.max())),
            40,
        )

        ax.hist(lam0, bins=bins, density=True, alpha=0.45,
                color="gray", label="epoch 0 (init)")
        ax.hist(lamT, bins=bins, density=True, alpha=0.55,
                color=color, label=f"epoch {final_epoch}")

        alpha_hat, xmin = final_alphas.get(lid, (None, None))
        if alpha_hat is not None and xmin is not None and lamT.max() > xmin:
            xs = np.logspace(np.log10(xmin), np.log10(lamT.max()), 100)
            C = (alpha_hat - 1) * xmin ** (alpha_hat - 1)
            ys = C * xs ** (-alpha_hat)
            ax.plot(xs, ys, "k--", linewidth=1.3,
                    label=fr"$\hat\alpha={alpha_hat:.2f}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Eigenvalue $\lambda$")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower left")

    for idx in range(len(roster), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if nrows > 1:
        for row in range(nrows):
            axes[row, 0].set_ylabel(r"Density $\rho(\lambda)$")
    else:
        axes_flat[0].set_ylabel(r"Density $\rho(\lambda)$")

    fig.suptitle(r"CNN empirical spectral distribution of $W W^{\!\top}$: initialization vs. trained",
                 y=1.00)
    fig.tight_layout()
    _save(fig, "fig_cnn_esd_comparison")


def main():
    history, eigs = load()
    plot_alpha_evolution(history)
    plot_esd_comparison(history, eigs)

    # Generate CNN figures only if the CNN outputs exist.
    cnn_tracking = RESULTS_DIR / "cnn_tracking.json"
    cnn_eigs = RESULTS_DIR / "cnn_eigenvalues.json"
    if cnn_tracking.exists() and cnn_eigs.exists():
        cnn_history, cnn_eigs_data = load_cnn()
        plot_cnn_alpha_evolution(cnn_history)
        plot_cnn_esd_comparison(cnn_history, cnn_eigs_data)
    else:
        print("cnn_tracking.json / cnn_eigenvalues.json not found, skipping CNN plots")


if __name__ == "__main__":
    main()
