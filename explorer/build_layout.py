"""Seriated sheet layouts: places each layer's neurons on its 96x96 grid so that neurons with similar weights sit
together. Embeds each neuron by its down_proj column ("writes": the direction it adds to the residual stream) or its
gate and up rows ("reads"), cosine-normalised, PCA to 50 dims, UMAP to 2D, then snaps to the grid: sort by y into
rows of 96, then by x within each row.

    uv run python explorer/build_layout.py --kind writes

Writes explorer/cache/layout_<kind>.npz: cell[L, N] (int16, row * 96 + col for neuron n of layer L).
"""
import argparse
import json
import math
import os
import re

import numpy as np
import umap
from huggingface_hub import snapshot_download
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
ap.add_argument("--kind", default="writes", choices=["writes", "reads"])
ap.add_argument("--pca", type=int, default=50)
ap.add_argument("--neighbors", type=int, default=30)
ap.add_argument("--out", default=None)
ap.add_argument("--layers", type=int, default=None)
a = ap.parse_args()

def model_slug(model_id):
    return model_id.lower().replace("/", "--")

try:
    root = snapshot_download(a.model, allow_patterns=["*.safetensors", "*.json"])
except Exception:
    root = snapshot_download(a.model, allow_patterns=["*.safetensors", "*.json"], local_files_only=True)

index_path = os.path.join(root, "model.safetensors.index.json")
if os.path.exists(index_path):
    index = json.load(open(index_path))["weight_map"]
    def tensor(name):
        with safe_open(os.path.join(root, index[name]), "pt") as f:
            return f.get_tensor(name).float().numpy()
    keys = set(index.keys())
else:
    st_file = os.path.join(root, "model.safetensors")
    def tensor(name):
        with safe_open(st_file, "pt") as f:
            return f.get_tensor(name).float().numpy()
    with safe_open(st_file, "pt") as f:
        keys = set(f.keys())

DOWN_RE = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.(?:mlp\.down_proj|feed_forward\.w2|shared_mlp\.output_linear)\.weight$"
)
down_keys = {int(m.group(1)): k for k in keys if (m := DOWN_RE.match(k))}
total_L = len(down_keys)
L = min(total_L, a.layers) if a.layers is not None else total_L
sample_down = tensor(down_keys[0])
N = sample_down.shape[1]
side = math.ceil(math.sqrt(N))



cells = np.zeros((L, N), np.int16)
for l in range(L):
    down_k = down_keys[l]
    if a.kind == "writes":
        X = tensor(down_k).T  # [N, hidden]
    else:
        prefix = down_k.rsplit(".", 2)[0] + "."
        if prefix + "w1.weight" in keys:
            gate = tensor(prefix + "w1.weight")
            up = tensor(prefix + "w3.weight")
            X = np.concatenate([gate, up], 1)
        elif prefix + "gate_proj.weight" in keys:
            gate = tensor(prefix + "gate_proj.weight")
            up = tensor(prefix + "up_proj.weight")
            X = np.concatenate([gate, up], 1)
        elif prefix.endswith(".shared_mlp.") and (prefix + "input_linear.weight") in keys:
            fused = tensor(prefix + "input_linear.weight")
            X = np.concatenate([fused[:N], fused[N:]], 1)
        else:
            raise KeyError(f"Missing gate/up weights for layer {l}")
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    pca_k = min(a.pca, X.shape[1], X.shape[0])
    Z = U[:, :pca_k] * S[:pca_k]
    xy = umap.UMAP(n_neighbors=min(a.neighbors, N - 1), min_dist=0.05, metric="cosine", random_state=0).fit_transform(Z)
    by_y = np.argsort(xy[:, 1], kind="stable")
    for r in range(side):
        start = r * side
        end = min((r + 1) * side, N)
        if start >= end:
            break
        band = by_y[start:end]
        band = band[np.argsort(xy[band, 0], kind="stable")]
        cells[l, band] = r * side + np.arange(len(band))
    print(f"L{l} done", flush=True)

out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", model_slug(a.model), f"layout_{a.kind}.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, cell=cells)
print("->", out)
