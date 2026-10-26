"""Per-neuron activation ranges for MLP layers of a model, plus their down-projection columns.

A neuron's activation is silu(gate) * up, the input to down_proj. Positive activations come from two sign regimes
(gate>0, up>0) and (gate<0, up<0); negative ones from (gate>0, up<0) and (gate<0, up>0). This records the extreme of
each regime per neuron over a mixed web + chat corpus, and saves W_down[:, n] for every neuron, which is the direction
a steering push on neuron n adds to the residual stream.

    uv run python sweep/neuron_stats.py --model LiquidAI/LFM2.5-350M --layers 0 --tokens 2000000

Writes stats/<slug>_L{layer}.npz with
    pp_max, nn_max   largest positive activation in the (+,+) and (-,-) regimes          [I]
    pn_min, np_min   most negative activation in the (gate+, up-) and (gate-, up+) regimes [I]
    abs_max          max |activation| over everything (the unit for steering doses)      [I]
    w_down           down_proj columns, one row per neuron                                [I, H] float16
    resid_norm_q10_50_90   quantiles of the per-token residual norm leaving the layer (scale for push sizes)
"""
import argparse
import itertools
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


def select_device_and_dtype():
    """Automatic device & dtype selection: prefer mps -> cuda -> cpu; bf16 on MPS/Ampere+, fp16 on Turing sm_75."""
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.bfloat16


def text_stream(tok, seed, local_text=None):
    if local_text:
        while True:
            yield local_text
    from datasets import load_dataset
    web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed)
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).shuffle(seed=seed)
    for w, c in zip(web, chat):
        yield w["text"]
        try:
            yield tok.apply_chat_template(c["messages"], tokenize=False, enable_thinking=False)
        except Exception:
            yield tok.apply_chat_template(c["messages"], tokenize=False)


def windows(tok, seq_len, batch, seed, local_text=None):
    """Packs documents (EOS-separated) into [batch, seq_len] token windows."""
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    buf = []
    chunk = []
    for text in text_stream(tok, seed, local_text):
        buf.extend(tok(text).input_ids + [eos])
        while len(buf) >= seq_len:
            chunk.extend(buf[:seq_len])
            buf = buf[seq_len:]
            if len(chunk) == batch * seq_len:
                yield torch.tensor(chunk).view(batch, seq_len)
                chunk = []


class RegimeStats:
    def __init__(self, n, device):
        f = lambda v: torch.full((n,), v, device=device, dtype=torch.float32)
        self.pp_max, self.nn_max = f(0.0), f(0.0)
        self.pn_min, self.np_min = f(0.0), f(0.0)
        self.gate = self.up = None
        self.resid_norms = []  # per-token residual norm leaving the layer, positions >= 1

    def update_resid(self, out):
        h = out[0] if isinstance(out, tuple) else out
        self.resid_norms.append(h[:, 1:].float().norm(dim=-1).flatten().cpu())

    def update(self, act):
        # position 0 of each window is an attention sink with outsized activations; it would inflate abs_max
        g, u, a = (t[:, 1:].flatten(0, 1).float() for t in (self.gate, self.up, act))
        gp, up_ = g > 0, u > 0
        ninf = torch.tensor(float("-inf"), device=a.device)
        pinf = torch.tensor(float("inf"), device=a.device)
        self.pp_max = torch.maximum(self.pp_max, torch.where(gp & up_, a, ninf).amax(0))
        self.nn_max = torch.maximum(self.nn_max, torch.where(~gp & ~up_, a, ninf).amax(0))
        self.pn_min = torch.minimum(self.pn_min, torch.where(gp & ~up_, a, pinf).amin(0))
        self.np_min = torch.minimum(self.np_min, torch.where(~gp & up_, a, pinf).amin(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--layers", default="0", help="comma-separated decoder layer indices")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Output base (default: stats/<slug>)")
    ap.add_argument("--text", default=None, help="Optional local text to avoid downloading web datasets")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]

    dev, dtype = select_device_and_dtype()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
    adapter = adapter_for(model, tok)
    lm = adapter.text_model
    lm.layers = adapter.layers()[: max(layers) + 1]  # nothing past the deepest target layer is needed

    stats = {}
    for l in layers:
        down_proj = adapter.down_proj(l)
        N = down_proj.in_features
        s = stats[l] = RegimeStats(N, dev)
        adapter.hook_mlp(l, s)
        lm.layers[l].register_forward_hook(lambda m, i, o, s=s: s.update_resid(o))

    n_steps = max(1, -(-a.tokens // (a.seq_len * a.batch)))
    t0 = time.time()
    with torch.inference_mode():
        for step, ids in enumerate(itertools.islice(windows(tok, a.seq_len, a.batch, a.seed, a.text), n_steps)):
            lm(input_ids=ids.to(dev), use_cache=False)
            if step % 10 == 0 or step == n_steps - 1:
                done = (step + 1) * a.seq_len * a.batch
                print(f"{step + 1}/{n_steps} steps, {done:,} tokens, {done / (time.time() - t0):,.0f} tok/s", flush=True)

    slug = model_slug(a.model)
    out_base = a.out or os.path.join(ROOT, "stats", slug)
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    for l, s in stats.items():
        pp, nn, pn, np_ = (x.cpu().numpy() for x in (s.pp_max, s.nn_max, s.pn_min, s.np_min))
        abs_max = np.maximum.reduce([pp, nn, -pn, -np_])
        w_down = adapter.down_proj(l).weight.detach().float().T.cpu().numpy().astype(np.float16)
        resid_q = np.quantile(torch.cat(s.resid_norms).numpy(), [0.1, 0.5, 0.9])
        path = f"{out_base}_L{l}.npz"
        np.savez(path, pp_max=pp, nn_max=nn, pn_min=pn, np_min=np_, abs_max=abs_max, w_down=w_down,
                 resid_norm_q10_50_90=resid_q, model=a.model, layer=l, tokens=n_steps * a.seq_len * a.batch)
        print(f"L{l}: residual norm leaving the layer p10/p50/p90 = {resid_q.round(1)}")
        dn_top = (nn > pp).mean()
        print(f"L{l}: abs_max median {np.median(abs_max):.3f} [{abs_max.min():.3f}, {abs_max.max():.3f}]; "
              f"(-,-) regime holds the positive max for {dn_top:.1%} of neurons -> {path}")


if __name__ == "__main__":
    main()
