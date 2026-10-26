"""Steer lens: what pushing each neuron does to the model's next-token prediction, through every layer after it.

For each neuron and sign, adds ±dose x (median residual norm / (mlp_scale * ||W_down[:, n]||)) to the neuron at every
position of a few short contexts, runs one forward pass, and compares the last position's log-probabilities with the
unsteered run. Keeps, per neuron and sign, the top tokens whose log-probability rises most (averaged over contexts)
and the KL divergence from the unsteered prediction.

    uv run python sweep/steer_lens.py --model LiquidAI/LFM2.5-350M --layers 0 --neurons 0-15

Writes and merges directly to explorer/cache/<slug>/steer_index.npz:
    up[L, N, k], down[L, N, k], up_kl[L, N], down_kl[L, N]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from explorer.models import adapter_for, model_slug

CHAT = ["Who are you?", "Write whatever you like."]
PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]


def select_device_and_dtype():
    """Automatic device & dtype selection: prefer mps -> cuda -> cpu; bf16 on MPS/Ampere+, fp16 on Turing sm_75."""
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.bfloat16


def parse_neurons(arg, total_neurons):
    """Parse subset of neurons: 'all', '0-15', '16' (first 16), or comma-separated."""
    if not arg or arg == "all":
        return list(range(total_neurons))
    if "-" in arg:
        lo, _, hi = arg.partition("-")
        return list(range(int(lo), int(hi or lo) + 1))
    if "," in arg:
        return [int(x) for x in arg.split(",") if x.strip()]
    k = int(arg)
    if k == 0:
        return [0]
    if k <= total_neurons:
        return list(range(k))
    return [k]


def parse_layers(arg, total_layers):
    """Parse layers argument: '0', '0-7', '0,2,4', or 'all'."""
    if not arg or arg == "all":
        return list(range(total_layers))
    if "," in arg:
        return [int(x) for x in arg.split(",") if x.strip()]
    if "-" in arg:
        lo, _, hi = arg.partition("-")
        return list(range(int(lo), int(hi or lo) + 1))
    return [int(arg)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--layers", default=None, help="e.g. '0', '0-7', or comma-separated")
    ap.add_argument("--neurons", default="all", help="Subset of neurons: 'all', '0-15', '16', or comma-separated")
    ap.add_argument("--dose", type=float, default=0.8)
    ap.add_argument("--neurons-per-batch", type=int, default=96)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default=None, help="Output path (default: explorer/cache/<slug>/steer_index.npz)")
    a = ap.parse_args()

    dev, dtype = select_device_and_dtype()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
    adapter = adapter_for(model, tok)
    layers = adapter.layers()
    L = len(layers)

    target_layers = parse_layers(a.layers, L)

    ctxs = []
    for c in CHAT:
        try:
            m = tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True,
                                        enable_thinking=False, tokenize=True)
        except Exception:
            m = tok.apply_chat_template([{"role": "user", "content": c}], add_generation_prompt=True, tokenize=True)
        ctxs.append(list(m["input_ids"] if hasattr(m, "keys") else m))
    for p in PLAIN:
        ctxs.append(list(tok(p).input_ids))

    C, T = len(ctxs), max(map(len, ctxs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id if tok.eos_token_id is not None else 0)
    ids = torch.tensor([[pad] * (T - len(c)) + c for c in ctxs], device=dev)
    mask = torch.tensor([[0] * (T - len(c)) + [1] * len(c) for c in ctxs], device=dev)
    posn = (mask.cumsum(-1) - 1).clamp(min=0)

    push = {"layer": None, "cols": None, "vals": None}
    resid = {}

    def make_hook(i):
        def hook(mod, inp):
            if push["layer"] != i:
                return
            h = inp[0].clone()
            rows = torch.arange(h.shape[0], device=h.device)
            h[rows, :, push["cols"]] += push["vals"][:, None].to(h.dtype)
            return (h,)
        return hook

    for i, layer_mod in enumerate(layers):
        down = adapter.down_proj(i)
        down.register_forward_pre_hook(make_hook(i))
        layer_mod.register_forward_hook(lambda m, inp, out, i=i: resid.__setitem__(i, (out[0] if isinstance(out, tuple) else out)))

    def last_logprobs(input_ids, attn, pos):
        out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=False, logits_to_keep=1)
        logits = out.logits[:, -1] if out.logits.dim() == 3 else out.logits
        return torch.log_softmax(logits.float(), -1)

    with torch.inference_mode():
        base = last_logprobs(ids, mask, posn)  # [C, V]
        norms = {i: r[mask.bool()].float().norm(dim=-1).median().item() for i, r in resid.items()}

    slug = model_slug(a.model)
    out_file = a.out
    if not out_file or not out_file.endswith(".npz"):
        out_file = os.path.join(ROOT, "explorer", "cache", slug, "steer_index.npz")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    sample_w = adapter.down_proj(0).weight
    N = sample_w.shape[1]

    # Load existing steer_index if present to merge
    up, down, up_kl, down_kl = None, None, None, None
    if os.path.exists(out_file):
        try:
            cur = dict(np.load(out_file))
            if ("up" in cur and "down" in cur and "up_kl" in cur and "down_kl" in cur and
                cur["up"].shape[:2] == (L, N) and cur["down"].shape[:2] == (L, N)):
                up = cur["up"].copy()
                down = cur["down"].copy()
                up_kl = cur["up_kl"].copy()
                down_kl = cur["down_kl"].copy()
        except Exception:
            pass

    if up is None:
        up = np.zeros((L, N, a.k), dtype=np.int32)
        down = np.zeros((L, N, a.k), dtype=np.int32)
        up_kl = np.zeros((L, N), dtype=np.float32)
        down_kl = np.zeros((L, N), dtype=np.float32)

    B = a.neurons_per_batch
    for l in target_layers:
        W = adapter.down_proj(l).weight  # [hidden, N]
        scale = adapter.mlp_scale(l)
        unit = norms[l] / (scale * W.float().norm(dim=0))  # [N]
        target_neurons = parse_neurons(a.neurons, N)

        out = {k: [] for k in ("up_tok", "down_tok", "up_kl", "down_kl")}
        t0 = time.time()
        for s in range(0, len(target_neurons), B):
            batch_indices = target_neurons[s: min(s + B, len(target_neurons))]
            n = torch.tensor(batch_indices, device=dev, dtype=torch.long)
            b = len(n)
            for sign, key in ((1.0, "up"), (-1.0, "down")):
                cols = n.repeat_interleave(C)
                vals = sign * a.dose * unit[cols]
                push.update(layer=l, cols=cols, vals=vals)
                with torch.inference_mode():
                    lp = last_logprobs(ids.repeat(b, 1), mask.repeat(b, 1), posn.repeat(b, 1)).view(b, C, -1)
                push["layer"] = None
                delta = (lp - base[None]).mean(1)  # [b, V]
                kl = (lp.exp() * (lp - base[None])).sum(-1).mean(1)  # KL(steered || base)
                out[f"{key}_tok"].append(delta.topk(a.k).indices.int().cpu())
                out[f"{key}_kl"].append(kl.cpu())

        layer_up_tok = torch.cat(out["up_tok"]).numpy()
        layer_down_tok = torch.cat(out["down_tok"]).numpy()
        layer_up_kl = torch.cat(out["up_kl"]).numpy()
        layer_down_kl = torch.cat(out["down_kl"]).numpy()

        up[l, target_neurons, :a.k] = layer_up_tok
        down[l, target_neurons, :a.k] = layer_down_tok
        up_kl[l, target_neurons] = layer_up_kl
        down_kl[l, target_neurons] = layer_down_kl

        print(f"L{l}: {len(target_neurons)} neurons in {time.time() - t0:.0f}s, KL+ median {np.median(layer_up_kl):.3f}", flush=True)

    np.savez(out_file, up=up, down=down, up_kl=up_kl, down_kl=down_kl)
    print(f"Wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
