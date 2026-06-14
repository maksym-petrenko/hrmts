"""
Generate thesis figures for the ViT and TinyGPT experiments.

Produces:
  - fig_vit_alpha_evolution.{pdf,png}  : per-projection alpha-hat over epochs
  - fig_vit_esd_comparison.{pdf,png}   : log-log ESD of representative Q / V / FFN1
  - fig_gpt_alpha_evolution.{pdf,png}  : same, over steps
  - fig_gpt_esd_comparison.{pdf,png}   : same, for TinyGPT

Mirrors the aesthetic of plots.py (serif font, log-log ESDs, MP/alpha=2 lines).
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
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# Consistent palette per projection type (Q/K/V/O/FFN1/FFN2/other).
KIND_COLOR = {
    "q":     "#1f77b4",
    "k":     "#17becf",
    "v":     "#d62728",
    "o":     "#9467bd",
    "fc1":   "#2ca02c",
    "fc2":   "#8c564b",
    "patch": "#bcbd22",
    "head":  "#7f7f7f",
    "tok_embed":  "#ff7f0e",
    "pos_embed":  "#e377c2",
    "other": "#555555",
}
KIND_LABEL = {
    "q":     "Q (attn)",
    "k":     "K (attn)",
    "v":     "V (attn)",
    "o":     "O (attn)",
    "fc1":   "FFN1",
    "fc2":   "FFN2",
    "patch": "patch embed",
    "head":  "classifier head",
    "tok_embed": "token embed",
    "pos_embed": "pos embed",
}


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
    if "patch_embed" in n:
        return "patch"
    if n == "head":
        return "head"
    return "other"


def _block_idx(longname: str):
    """Extract block index from names like 'blocks.3.attn.q' or 'blocks.3.mlp.fc1'."""
    parts = longname.split(".")
    if len(parts) >= 2 and parts[0] == "blocks":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def load(track_file, eig_file):
    with open(RESULTS_DIR / track_file) as f:
        history = json.load(f)
    with open(RESULTS_DIR / eig_file) as f:
        eigs = {int(k): v for k, v in json.load(f).items()}
    return history, eigs


# ---------------------------------------------------------------------------
# Alpha-over-time panel
# ---------------------------------------------------------------------------
def plot_alpha_evolution(history, x_key, title, outname, xlabel, metric_key="test_acc",
                          metric_label="Test accuracy", include_kinds=None):
    """
    One axes per projection kind, lines = (block index, color = kind).
    Bottom panel shows metric_key vs x_key (test accuracy or val loss).
    """
    xs = [r[x_key] for r in history]

    # Collect (kind, block_idx) -> list of alpha across time
    series = {}
    for rec in history:
        for lm in rec["layers"]:
            kind = _kind(lm["longname"])
            if include_kinds is not None and kind not in include_kinds:
                continue
            block = _block_idx(lm["longname"])
            series.setdefault((kind, block, lm["longname"]), []).append(lm["alpha"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0]},
    )

    # Group by kind for coloring; vary alpha across block depth.
    by_kind = {}
    for (kind, block, long), ys in series.items():
        by_kind.setdefault(kind, []).append((block, long, ys))

    kind_order = [k for k in ("q", "k", "v", "o", "fc1", "fc2", "patch", "head")
                  if k in by_kind]

    for kind in kind_order:
        entries = sorted(by_kind[kind], key=lambda e: (e[0] if e[0] is not None else -1))
        n = len(entries)
        base_color = KIND_COLOR[kind]
        for i, (block, long, ys) in enumerate(entries):
            # Vary line transparency by block index for visual continuity.
            a = 0.45 + 0.55 * (i / max(1, n - 1)) if n > 1 else 0.85
            # Sanitize: cap off-scale alpha (common at init).
            ys_plot = [y if (y is not None and y < 12) else np.nan for y in ys]
            label = KIND_LABEL[kind] if i == 0 else None
            ax1.plot(xs, ys_plot, marker="o", markersize=3.2, linewidth=1.25,
                     color=base_color, alpha=a, label=label)

    ax1.axhline(2.0, color="gray", linestyle=":", linewidth=1)
    ax1.axhline(4.0, color="gray", linestyle=":", linewidth=1)
    ax1.text(xs[-1], 2.05, r"$\alpha=2$ (very heavy-tailed)",
             fontsize=8, color="gray", ha="right", va="bottom")
    ax1.text(xs[-1], 4.05, r"$\alpha=4$ (MP bulk)",
             fontsize=8, color="gray", ha="right", va="bottom")

    ax1.set_ylim(1.3, 7.5)
    ax1.set_ylabel(r"Power-law exponent $\hat\alpha$")
    ax1.legend(loc="upper right", ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    ys_metric = [r.get(metric_key) for r in history]
    ax2.plot(xs, ys_metric, marker="s", color="black", markersize=3.5, linewidth=1.2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(metric_label)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{outname}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ESD comparison
# ---------------------------------------------------------------------------
def plot_esd_comparison(history, eigs, representatives, final_key, title, outname):
    """
    representatives: list of (longname, label, color).
    final_key: which epoch/step key in `eigs` to treat as 'final'.
    """
    # Map longname -> (alpha, xmin) from the final history record.
    final_record = history[-1]
    final_params = {
        lm["longname"]: (lm["alpha"], lm.get("xmin"))
        for lm in final_record["layers"]
    }

    n = len(representatives)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows))
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.atleast_1d(axes).flatten()

    init_eigs = eigs[0]
    final_eigs = eigs[final_key]

    for ax, (longname, label, color) in zip(axes_flat, representatives):
        if longname not in init_eigs or longname not in final_eigs:
            ax.set_visible(False)
            continue
        lam0 = np.array(init_eigs[longname])
        lamT = np.array(final_eigs[longname])
        if lam0.size == 0 or lamT.size == 0:
            ax.set_visible(False)
            continue

        lo = min(lam0.min(), lamT.min())
        hi = max(lam0.max(), lamT.max())
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)

        ax.hist(lam0, bins=bins, density=True, alpha=0.45,
                color="gray", label="init")
        ax.hist(lamT, bins=bins, density=True, alpha=0.55,
                color=color, label="trained")

        alpha_hat, xmin = final_params.get(longname, (None, None))
        if alpha_hat is not None and xmin is not None and xmin > 0 and alpha_hat > 1.0:
            xs = np.logspace(np.log10(xmin), np.log10(lamT.max()), 100)
            C = (alpha_hat - 1) * xmin ** (alpha_hat - 1)
            ys = C * xs ** (-alpha_hat)
            ax.plot(xs, ys, "k--", linewidth=1.3,
                    label=fr"$\hat\alpha={alpha_hat:.2f}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Eigenvalue $\lambda$")
        ax.set_ylabel(r"Density $\rho(\lambda)$")
        ax.set_title(label)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower left")

    for idx in range(len(representatives), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, y=1.00)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"{outname}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _pick_representative(history, kind, prefer_middle=True):
    """Return a longname of a layer with the given kind, preferring a middle block."""
    candidates = set()
    for rec in history:
        for lm in rec["layers"]:
            if _kind(lm["longname"]) == kind:
                candidates.add(lm["longname"])
    if not candidates:
        return None
    cands = sorted(candidates)
    if prefer_middle and len(cands) >= 2:
        return cands[len(cands) // 2]
    return cands[0]


def do_vit():
    track = RESULTS_DIR / "vit_tracking.json"
    eig = RESULTS_DIR / "vit_eigenvalues.json"
    if not track.exists() or not eig.exists():
        print(f"[vit] results missing at {track} / {eig}, skip")
        return
    history, eigs = load("vit_tracking.json", "vit_eigenvalues.json")

    plot_alpha_evolution(
        history, "epoch",
        "ViT on CIFAR-10: per-projection spectral evolution",
        "fig_vit_alpha_evolution",
        xlabel="Epoch", metric_key="test_acc", metric_label="Test accuracy",
        include_kinds={"q", "k", "v", "o", "fc1", "fc2", "patch", "head"},
    )

    reps = []
    for kind in ("q", "v", "fc1", "fc2", "o"):
        long = _pick_representative(history, kind)
        if long is not None:
            reps.append((long, f"{KIND_LABEL[kind]}  [{long}]", KIND_COLOR[kind]))
    # Keep to <= 6 panels
    reps = reps[:6]

    final_key = max(eigs.keys())
    plot_esd_comparison(
        history, eigs, reps, final_key,
        r"ViT: ESD of $WW^{\!\top}$ at init vs trained",
        "fig_vit_esd_comparison",
    )


def do_gpt():
    track = RESULTS_DIR / "gpt_tracking.json"
    eig = RESULTS_DIR / "gpt_eigenvalues.json"
    if not track.exists() or not eig.exists():
        print(f"[gpt] results missing at {track} / {eig}, skip")
        return
    history, eigs = load("gpt_tracking.json", "gpt_eigenvalues.json")

    plot_alpha_evolution(
        history, "step",
        "TinyGPT on Tiny Shakespeare: per-projection spectral evolution",
        "fig_gpt_alpha_evolution",
        xlabel="Training step", metric_key="val_loss", metric_label="Val loss",
        include_kinds={"q", "k", "v", "o", "fc1", "fc2", "head"},
    )

    reps = []
    for kind in ("q", "v", "fc1", "fc2", "o"):
        long = _pick_representative(history, kind)
        if long is not None:
            reps.append((long, f"{KIND_LABEL[kind]}  [{long}]", KIND_COLOR[kind]))
    reps = reps[:6]
    final_key = max(eigs.keys())
    plot_esd_comparison(
        history, eigs, reps, final_key,
        r"TinyGPT: ESD of $WW^{\!\top}$ at init vs trained",
        "fig_gpt_esd_comparison",
    )


if __name__ == "__main__":
    do_vit()
    do_gpt()
