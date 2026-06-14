import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
import weightwatcher as ww

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def main():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    d = ww.WeightWatcher(model=m).analyze(plot=False, randomize=False, mp_fit=False)
    alphas = {}
    for _, r in d.iterrows():
        key = str(r.get("longname", "")) or f"layer_{int(r['layer_id'])}"
        alphas[key] = {
            "alpha": float(r["alpha"]),
            "layer_id": int(r["layer_id"]),
            "M": int(r["M"]),
            "N": int(r["N"]),
            "longname": key,
        }
        print(f"{key:35s} alpha {float(r['alpha']):.2f}")

    evs = {}
    for n, p in m.named_parameters():
        if "weight" not in n or p.ndim < 2:
            continue
        W = p.detach().cpu().numpy()
        if W.ndim == 4:
            W = W.reshape(W.shape[0], -1)
        if min(W.shape) < 10:
            continue
        sv = np.linalg.svd(W, compute_uv=False)
        evs[n] = (sv ** 2).tolist()

    (OUT / "resnet_alpha.json").write_text(json.dumps({"alphas": alphas, "evs": evs}))


if __name__ == "__main__":
    main()
