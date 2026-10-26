"""Direct-path connections for every neuron in every layer, for the explorer's details tab.

For layers i < j, P[B, A] = (W_gate_j[B] . v_A) * (W_up_j[B] . v_A) with v_A = (1 + w_j) * W_down_i[:, A], w_j the
pre-MLP norm weight of layer j. Neuron A's downstream links in layer j are column A of P, |P| z-scored over B; neuron
B's upstream links in layer i are row B of P, |P| z-scored over A. Each neuron keeps its k strongest links per side
across all layers, with sign(P).

    uv run python explorer/build_connections.py            # CUDA, about 5 min on a 4090

Writes explorer/cache/connections.npz: {down,up}_layer [L, N, k] int8, {down,up}_n int16, {down,up}_z float16,
{down,up}_sign int8.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
ap.add_argument("--k", type=int, default=24)
ap.add_argument("--out", default=None)
ap.add_argument("--layers", type=int, default=None)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

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
            return f.get_tensor(name).to(dev, torch.bfloat16)
    keys = set(index.keys())
else:
    st_file = os.path.join(root, "model.safetensors")
    def tensor(name):
        with safe_open(st_file, "pt") as f:
            return f.get_tensor(name).to(dev, torch.bfloat16)
    with safe_open(st_file, "pt") as f:
        keys = set(f.keys())

DOWN_RE = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.(?:mlp\.down_proj|feed_forward\.w2|shared_mlp\.output_linear)\.weight$"
)
down_keys = {int(m.group(1)): k for k in keys if (m := DOWN_RE.match(k))}
total_L = len(down_keys)
L = min(total_L, a.layers) if a.layers is not None else total_L

down, gate, up, norm = [], [], [], []
for l in range(L):
    down_k = down_keys[l]
    d_t = tensor(down_k)
    down.append(d_t)
    prefix = down_k.rsplit(".", 2)[0] + "."
    if prefix + "w1.weight" in keys:
        g_t = tensor(prefix + "w1.weight")
        u_t = tensor(prefix + "w3.weight")
    elif prefix + "gate_proj.weight" in keys:
        g_t = tensor(prefix + "gate_proj.weight")
        u_t = tensor(prefix + "up_proj.weight")
    elif prefix.endswith(".shared_mlp.") and (prefix + "input_linear.weight") in keys:
        fused = tensor(prefix + "input_linear.weight")
        N_l = d_t.shape[1]
        g_t, u_t = fused[:N_l], fused[N_l:]
    else:
        raise KeyError(f"Missing gate/up weights for layer {l}")
    gate.append(g_t)
    up.append(u_t)

    norm_k_lfm = f"model.layers.{l}.ffn_norm.weight"
    norm_k_qwen_cond = f"model.language_model.layers.{l}.post_attention_layernorm.weight"
    norm_k_causal = f"model.layers.{l}.post_attention_layernorm.weight"
    if norm_k_lfm in keys:
        norm.append(tensor(norm_k_lfm).float().to(torch.bfloat16))
    elif norm_k_qwen_cond in keys or (norm_k_causal in keys and "mlp.down_proj.weight" in down_k):
        nk = norm_k_qwen_cond if norm_k_qwen_cond in keys else norm_k_causal
        norm.append((1.0 + tensor(nk).float()).to(torch.bfloat16))
    elif norm_k_causal in keys:
        norm.append(tensor(norm_k_causal).float().to(torch.bfloat16))
    else:
        raise KeyError(f"Missing norm weight for layer {l}")

N = down[0].shape[1]
K = a.k
# running top-k per neuron and side: score, source layer, source neuron, sign
best = {s: (torch.full((L, N, K), -1e9, device=dev), torch.zeros((L, N, K), dtype=torch.int16, device=dev),
            torch.zeros((L, N, K), dtype=torch.int16, device=dev), torch.zeros((L, N, K), dtype=torch.int8, device=dev))
        for s in ("down", "up")}


def merge(side, l, z, idx, other_layer, sign):
    bz, bl, bn, bs = best[side]
    cz = torch.cat([bz[l], z], 1)
    cl = torch.cat([bl[l], torch.full_like(idx, other_layer, dtype=torch.int16)], 1)
    cn = torch.cat([bn[l], idx.to(torch.int16)], 1)
    cs = torch.cat([bs[l], sign], 1)
    top = cz.topk(K, dim=1)
    bz[l], bl[l], bn[l], bs[l] = top.values, cl.gather(1, top.indices), cn.gather(1, top.indices), cs.gather(1, top.indices)


t0 = time.time()
with torch.inference_mode():
    for j in range(1, L):
        for i in range(j):
            V = down[i] * norm[j][:, None]  # [hidden, N_i]
            P = (gate[j] @ V).float() * (up[j] @ V).float()  # [N_j (B), N_i (A)]
            M = P.abs()
            # downstream of A (layer i): column A over B
            zc = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-8)
            top = zc.topk(K, dim=0)  # [K, N_i]
            sgn = torch.where(P.gather(0, top.indices) > 0, 1, -1).to(torch.int8)
            merge("down", i, top.values.T, top.indices.T, j, sgn.T)
            # upstream of B (layer j): row B over A
            zr = (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-8)
            top = zr.topk(K, dim=1)  # [N_j, K]
            sgn = torch.where(P.gather(1, top.indices) > 0, 1, -1).to(torch.int8)
            merge("up", j, top.values, top.indices, i, sgn)
            del P, M, zc, zr
        print(f"L{j} done ({time.time() - t0:.0f}s)", flush=True)

out = {}
for side, (bz, bl, bn, bs) in best.items():
    out[f"{side}_z"] = bz.clamp(min=-60000).cpu().numpy().astype(np.float16)
    out[f"{side}_layer"] = bl.cpu().numpy().astype(np.int8)
    out[f"{side}_n"] = bn.cpu().numpy()
    out[f"{side}_sign"] = bs.cpu().numpy()
out_path = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", model_slug(a.model), "connections.npz")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez_compressed(out_path, **out, k=K)
print("->", out_path, f"{os.path.getsize(out_path) / 1e6:.0f} MB")
