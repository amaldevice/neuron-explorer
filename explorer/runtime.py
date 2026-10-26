"""Live Qwen3.5 model for the explorer: steered generation (add or multiply one neuron, per batch row), per-token
neuron activations on arbitrary text, and logit-lens tokens for a neuron's down column.

The hook sits on the input of a layer's down_proj, where the neuron activations silu(gate) * up live, so add,
multiply, and reading are all exact. Add doses are in units of the median residual norm leaving the layer, the same
unit as the sweep's `--dose-unit rel`.
"""
import json
import os
import sqlite3
import threading

import numpy as np
import torch
from transformers import AutoTokenizer

try:
    from models import MODEL as DEFAULT_MODEL, adapter_for, arch_from_model_or_config, load_model, model_slug
except ImportError:
    from explorer.models import MODEL as DEFAULT_MODEL, adapter_for, arch_from_model_or_config, load_model, model_slug

MODEL = os.environ.get("EXPLORER_MODEL", DEFAULT_MODEL)
NO_MODEL = bool(os.environ.get("EXPLORER_NO_MODEL"))  # cache-only: never download or load weights
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SLUG = model_slug(MODEL)
MODEL_CACHE = os.path.join(CACHE, SLUG)
AGENT_CACHE = os.path.join(CACHE, "agent")


class Runtime:
    def __init__(self, stats_dir):
        self.stats_dir = stats_dir
        self.arch = arch_from_model_or_config(MODEL)
        os.makedirs(MODEL_CACHE, exist_ok=True)
        self._db = sqlite3.connect(os.path.join(MODEL_CACHE, "connections.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS con (key TEXT PRIMARY KEY, val TEXT)")
        self._db_lock = threading.Lock()
        self._lens_idx = None
        self._con_idx = None
        self.lock = threading.Lock()
        self.model = self.tok = self.adapter = None
        self.status = "idle"
        self.resid_p50 = {}
        self._push = None  # [(layer, neuron, mode, per-row values tensor)], any number of neurons at once (neuron None = all)
        self._heads = None  # [(layer, head, scale)]: attention heads scaled (0 = ablated) during a read; head None = all
        self._read = None  # (layer, neuron) while reading
        self._seen = {}
        self._con_cache = {}
        self._capture_all = None
        self._capture_from = 0
        self._capture_full = None
        self._lens_cache = {}

    # ---------- loading ----------
    def ensure_loaded(self):
        if self.model is not None:
            return
        if NO_MODEL:
            raise RuntimeError("live model disabled (EXPLORER_NO_MODEL)")
        with self.lock:
            if self.model is not None:
                return
            self.status = "loading"
            dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(MODEL)
            self.model = load_model(MODEL)
            self.dev = dev
            self.adapter = adapter_for(self.model, self.tok)
            self.lm = self.adapter.text_model
            for i, layer in enumerate(self.adapter.layers()):
                self.adapter.down_proj(layer).register_forward_pre_hook(self._hook(i))
                layer.register_forward_hook(self._resid_hook(i))
                proj = self.adapter.mixer_out_proj(layer)
                heads = self.adapter.n_heads(layer)
                proj.register_forward_pre_hook(self._head_hook(i, heads))
            self.status = "ready"

    def _hook(self, i):
        def hook(mod, inp):
            h = inp[0]
            if self._capture_full is not None:
                self._capture_full[i] = h[0].float().cpu()  # [tokens, inter]
            if self._capture_all is not None:
                a = h[:, self._capture_from:].float().flatten(0, 1)  # [tokens, inter]
                mx, amx = a.max(0)
                mn, amn = a.min(0)
                self._capture_all[i] = (mx.cpu(), amx.cpu(), mn.cpu(), amn.cpu())
            if self._read is not None and self._read[0] == i:
                n = self._read[1]
                self._seen["act"] = h[0, :, n].float().cpu()
            mine = [p for p in self._push if p[0] == i] if self._push else []
            if mine:
                h = h.clone()
                for _, n, mode, vals in mine:
                    if n is None:  # the whole layer's MLP (multiply only)
                        h = h * vals[:, None, None].to(h.dtype)
                    elif mode == "mul":
                        h[:, :, n] = h[:, :, n] * vals[:, None].to(h.dtype)
                    else:
                        h[:, :, n] = h[:, :, n] + vals[:, None].to(h.dtype)
                return (h,)
        return hook

    def _head_hook(self, i, n_heads):
        def hook(mod, inp):
            mine = [x for x in self._heads if x[0] == i] if self._heads else []
            if not mine:
                return None
            h = inp[0].clone()
            v = h.view(*h.shape[:-1], n_heads, h.shape[-1] // n_heads)
            for _, head, scale in mine:
                if head is None:
                    v.mul_(scale)
                else:
                    v[..., head, :] *= scale
            return (h,)
        return hook

    def n_heads(self, layer):
        return self.adapter.n_heads(self.lm.layers[layer])

    def _resid_hook(self, i):
        def hook(mod, inp, out):
            if self._seen.get("want_resid") == i:
                h = out[0] if isinstance(out, tuple) else out
                self._seen["resid"] = h[0, 1:].float().norm(dim=-1).cpu()
        return hook

    def layer_resid_p50(self, layer):
        if layer in self.resid_p50:
            return self.resid_p50[layer]
        slug = model_slug(MODEL)
        path = os.path.join(self.stats_dir, f"{slug}_L{layer}.npz")
        if not os.path.exists(path) and MODEL == "Qwen/Qwen3.5-4B":
            path = os.path.join(self.stats_dir, f"q35-4b_L{layer}.npz")
        if os.path.exists(path):
            v = float(np.load(path)["resid_norm_q10_50_90"][1])
        else:
            ids = self.tok(REFERENCE_TEXT, return_tensors="pt").input_ids.to(self.dev)
            self._seen["want_resid"] = layer
            with torch.inference_mode():
                self.lm(input_ids=ids, use_cache=False)
            self._seen.pop("want_resid")
            v = float(self._seen.pop("resid").median())
        self.resid_p50[layer] = v
        return v

    def down_col(self, layer, n):
        return self.adapter.down_proj(self.lm.layers[layer]).weight[:, n].detach().float()

    # ---------- steering ----------
    def chat_ids(self, prompt):
        return self.messages_ids([{"role": "user", "content": prompt}])

    def messages_ids(self, messages):
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        return list(enc["input_ids"] if hasattr(enc, "keys") else enc)

    def steer_stream(self, layer, n, rows_messages, dose, mode, max_new=150, cancel=None):
        """Greedy-decodes three conversations in one batch: row 0 pushed up, row 1 unsteered, row 2 pushed down.
        rows_messages holds each row's own message list (the rows can differ in length; they are left-padded).
        Yields (row, text_delta), then (None, None)."""
        yield from self.steer_multi([(layer, n, dose, mode)], rows_messages, max_new, cancel)

    def steer_multi(self, pushes, rows_messages, max_new=150, cancel=None):
        """steer_stream for any number of neurons at once: pushes = [(layer, n, dose, mode)], all applied together.
        Each is reversed in row 2 the same way a single neuron is (add: -dose; mul: x -dose)."""
        self.ensure_loaded()
        with self.lock:
            if cancel is not None and cancel.is_set():
                yield None, None
                return
            try:
                push = [self._push_entry(layer, n, dose, mode, (1, 0, -1)) for layer, n, dose, mode in pushes]
                seqs = [self.messages_ids(m) for m in rows_messages]
                T = max(map(len, seqs))
                pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                ids = torch.tensor([[pad] * (T - len(q)) + q for q in seqs], device=self.dev)
                mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in seqs], device=self.dev)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                eos = self.adapter.eos_token_id()
                done, out_ids, sent = [False] * 3, [[] for _ in range(3)], [""] * 3
                self._push = push
                with torch.inference_mode():
                    out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True,
                                     logits_to_keep=1)
                    for _ in range(max_new):
                        if cancel is not None and cancel.is_set():
                            break
                        nxt = out.logits[:, -1].argmax(-1)
                        for r in range(3):
                            if done[r]:
                                continue
                            t = int(nxt[r])
                            if t == eos:
                                done[r] = True
                                continue
                            out_ids[r].append(t)
                            text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                            if not text.endswith("\ufffd"):
                                yield r, text[len(sent[r]):]
                                sent[r] = text
                        if all(done):
                            break
                        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
                        pos = pos[:, -1:] + 1
                        out = self.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                                         past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            finally:
                self._push = None
        yield None, None

    def token_units(self, ids):
        """Groups token ids into displayable units: byte-level pieces of one multi-byte character (emoji, CJK split
        across tokens) join the token that completes it. Returns [(text, [token indices])]."""
        units, pending, prev = [], [], ""
        for i in range(len(ids)):
            full = self.tok.decode(ids[: i + 1])
            pending.append(i)
            if full.endswith("\ufffd") and i < len(ids) - 1:
                continue
            units.append((full[len(prev):] if full.startswith(prev) else self.tok.decode([ids[j] for j in pending]), pending))
            prev, pending = full, []
        # keep emoji sequences whole: joiners, variation selectors and skin tones attach to the unit before them
        glue = {"\u200d", "\ufe0f", "\ufe0e", "\u20e3"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
        merged = []
        for text, idx in units:
            if merged and text and (text[0] in glue or merged[-1][0].endswith("\u200d")):
                merged[-1] = (merged[-1][0] + text, merged[-1][1] + idx)
            else:
                merged.append((text, idx))
        return merged

    # ---------- reading ----------
    def _push_entry(self, layer, n, dose, mode, rows):
        """One neuron's push for the down_proj hook, per batch row: rows holds +1 (pushed), 0 (left alone) or -1
        (pushed the other way). add doses are in units of the layer's median residual norm along the neuron's down
        column; mul doses multiply the activation (left alone = x1, the other way = x -dose)."""
        if n is None and mode != "mul":
            raise ValueError("a whole-layer MLP push must be mode 'mul' (e.g. dose 0 to ablate)")
        if mode == "mul":
            vals = [dose if r > 0 else 1.0 if r == 0 else -dose for r in rows]
        else:
            scale = self.adapter.mlp_scale(self.lm.layers[layer])
            unit = self.layer_resid_p50(layer) / (scale * float(self.down_col(layer, n).norm()))
            vals = [r * dose * unit for r in rows]
        return (layer, n, mode, torch.tensor(vals, device=self.dev))

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        """Per-token activation of neuron n at `layer` on `text` (as a user turn when chat=True), optionally with other
        neurons pushed at the same time: pushes = [(layer, n, dose, mode)], the same units as steering (n None = the
        layer's whole MLP, mul only), and attention heads scaled: heads = [(layer, head, scale)], head None = all heads,
        scale 0 = ablated. Only earlier layers can move the read (a same-layer MLP push lands after the capture; a
        same-layer head change lands before it)."""
        self.ensure_loaded()
        with self.lock:
            if chat:
                ids = self.chat_ids(text)
            else:
                ids = self.tok(text).input_ids
            self._read = (layer, n)
            self._push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
            self._heads = list(heads) or None if heads else None
            try:
                with torch.inference_mode():
                    self.lm(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
            finally:
                self._read = None
                self._push = None
                self._heads = None
            act = self._seen.pop("act").tolist()
        units = []
        for text, idx in self.token_units(ids):
            vals = [act[i] for i in idx]
            units.append({"text": text, "act": max(vals, key=abs), "pieces": vals})
        return {"units": units}

    def next_logprobs(self, text, chat=True, prefix="", pushes=None, heads=None):
        """Log-probabilities of the next token after `text` (as a user turn when chat=True, so the next token opens the
        reply) followed by `prefix`, as a [vocab] tensor. With pushes / heads (same units as read()) it also returns
        the same distribution under that intervention: (base, intervened or None)."""
        self.ensure_loaded()
        ids = self.chat_ids(text) if chat else self.tok(text).input_ids
        if prefix:
            ids = ids + self.tok(prefix, add_special_tokens=False).input_ids

        def run(push, hd):
            with self.lock:
                self._push, self._heads = push, hd
                try:
                    with torch.inference_mode():
                        out = self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=False, logits_to_keep=1)
                finally:
                    self._push = self._heads = None
                return torch.log_softmax(out.logits[0, -1].float(), -1).cpu()

        base = run(None, None)
        if not pushes and not heads:
            return base, None
        push = [self._push_entry(pl, pn, d, m, (1,)) for pl, pn, d, m in pushes or []] or None
        return base, run(push, list(heads) if heads else None)

    def con_index(self):
        """Precomputed connections (build_connections.py), loaded once."""
        if self._con_idx is None:
            path = os.path.join(MODEL_CACHE, "connections.npz")
            if os.path.exists(path):
                try:
                    z = np.load(path)
                    expected = (self.arch["n_layers"], self.arch["inter"])
                    if z["down_z"].shape[:2] == expected and z["up_z"].shape[:2] == expected:
                        self._con_idx = {key: z[key] for key in z.files}
                        self._con_idx["k"] = int(z["k"])
                    else:
                        self._con_idx = False
                except Exception:
                    self._con_idx = False
            else:
                self._con_idx = False
        return self._con_idx or None

    # ---------- logit lens ----------
    def lens_index(self):
        if self._lens_idx is None:
            path = os.path.join(MODEL_CACHE, "lens_index.npz")
            if os.path.exists(path):
                try:
                    z = np.load(path)
                    expected = (self.arch["n_layers"], self.arch["inter"])
                    if z["up"].shape[:2] == expected and z["down"].shape[:2] == expected:
                        self._lens_idx = {"up": z["up"], "down": z["down"]}
                    else:
                        self._lens_idx = None
                except Exception:
                    self._lens_idx = None
        return self._lens_idx

    def steer_index(self):
        if getattr(self, "_steer_idx", None) is None:
            path = os.path.join(MODEL_CACHE, "steer_index.npz")
            if os.path.exists(path):
                try:
                    z = np.load(path)
                    if z["up"].shape[:2] == (self.arch["n_layers"], self.arch["inter"]):
                        self._steer_idx = dict(z)
                    else:
                        self._steer_idx = False
                except Exception:
                    self._steer_idx = False
            else:
                self._steer_idx = False
        return self._steer_idx or None

    def lens(self, layer, n, k=16):
        idx = self.lens_index()
        if idx is not None and k <= idx["up"].shape[-1]:
            v = self.vocab_strings()
            out = {"up": [v[t] for t in idx["up"][layer, n, :k]], "down": [v[t] for t in idx["down"][layer, n, :k]]}
            st = self.steer_index()
            if st is not None:
                out.update(steer_up=[v[t] for t in st["up"][layer, n, :k]], steer_down=[v[t] for t in st["down"][layer, n, :k]],
                           kl_up=round(float(st["up_kl"][layer, n]), 3), kl_down=round(float(st["down_kl"][layer, n]), 3))
            return out
        self.ensure_loaded()
        if (layer, n, k) in self._lens_cache:
            return self._lens_cache[(layer, n, k)]
        with self.lock, torch.inference_mode():
            d = self.down_col(layer, n).to(self.model.dtype)[None, None]
            out = {}
            for sign, name in ((1, "up"), (-1, "down")):
                logits = self.model.lm_head(self.adapter.final_norm()(sign * d))[0, 0].float()
                top = logits.topk(k).indices.tolist()
                out[name] = [self.tok.decode([t]) for t in top]
        self._lens_cache[(layer, n, k)] = out
        return out

    # ---------- connections ----------
    def connections(self, layer, n, k=24, lens_k=5):
        """Direct-path weights between neuron (layer, n) and neurons in other layers, ignoring everything between.
        Downstream B in layer j > layer: g = W_gate_j[B] . ((1 + w_j) * d_n), u = W_up_j[B] . (...), where d_n is this
        neuron's down column and w_j the pre-MLP norm weight. Upstream A in layer i < layer: the same with A's down
        column against this neuron's gate and up rows. Score = |g * u| z-scored within each layer; sign = sign(g * u).
        Each returned neuron carries its own logit-lens tokens."""
        key = (layer, n, k, lens_k)
        if key in self._con_cache:
            return self._con_cache[key]
        ci = self.con_index()
        if ci is not None and k <= ci["k"]:
            out = {}
            for side in ("down", "up"):
                z = ci[f"{side}_z"][layer, n]
                out[side] = [{"layer": int(ci[f"{side}_layer"][layer, n, i]), "n": int(ci[f"{side}_n"][layer, n, i]),
                              "z": round(float(z[i]), 2), "sign": int(ci[f"{side}_sign"][layer, n, i])}
                             for i in range(k) if z[i] > -1e4]
            idx = self.steer_index() or self.lens_index()  # linked neurons show their steer lens when there is one
            if lens_k and idx is not None:
                v = self.vocab_strings()
                lg = self.lens_index()
                for c in out["down"] + out["up"]:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
                    if lg is not None and lg is not idx:  # the logit lens too, shown faintly under the steer lens
                        c["logit"] = [v[t] for t in lg["up"][c["layer"], c["n"], :lens_k]]
                        c["logit_neg"] = [v[t] for t in lg["down"][c["layer"], c["n"], :lens_k]]
            return out
        dkey = f"{layer}:{n}:{k}:{lens_k}"
        with self._db_lock:
            row = self._db.execute("SELECT val FROM con WHERE key = ?", (dkey,)).fetchone()
        if row:
            out = json.loads(row[0])
            self._con_cache[key] = out
            return out
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            dt = self.model.dtype
            out = {"down": [], "up": []}
            d = self.down_col(layer, n).to(dt)
            me_l = self.lm.layers[layer]
            scale_me = self.adapter.pre_mlp_norm_scale(me_l).to(dt)
            g_w, u_w = self.adapter.gate_up_weights(me_l)
            r_g = g_w[n].to(dt) * scale_me
            r_u = u_w[n].to(dt) * scale_me
            for j, Lyr in enumerate(self.lm.layers):
                if j == layer:
                    continue
                if j > layer:
                    v = d * self.adapter.pre_mlp_norm_scale(Lyr).to(dt)
                    gj_w, uj_w = self.adapter.gate_up_weights(Lyr)
                    g, u = (gj_w.to(dt) @ v).float(), (uj_w.to(dt) @ v).float()
                    side = "down"
                else:
                    W = self.adapter.down_proj(Lyr).weight.to(dt)
                    g, u = (r_g @ W).float(), (r_u @ W).float()
                    side = "up"
                prod = g * u
                mag = prod.abs()
                z = (mag - mag.mean()) / (mag.std() + 1e-8)
                top = z.topk(min(k, z.numel()))
                out[side] += [{"layer": j, "n": int(i), "z": round(float(sc), 2), "sign": 1 if prod[i] > 0 else -1}
                              for sc, i in zip(top.values.tolist(), top.indices.tolist())]
            for side in out:
                out[side] = sorted(out[side], key=lambda c: -c["z"])[:k]
            every = out["down"] + out["up"]
            idx = self.steer_index() or self.lens_index()
            if every and lens_k and idx is not None and lens_k <= idx["up"].shape[-1]:
                v = self.vocab_strings()
                for c in every:
                    c["lens"] = [v[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [v[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
            elif every and lens_k:
                D = torch.stack([self.adapter.down_proj(self.lm.layers[c["layer"]]).weight[:, c["n"]] for c in every])
                logits = self.model.lm_head(self.adapter.final_norm()(D[:, None]))[:, 0].float()
                ids, ids_neg = logits.topk(lens_k).indices.tolist(), (-logits).topk(lens_k).indices.tolist()
                for c, row, row_neg in zip(every, ids, ids_neg):
                    c["lens"] = [self.tok.decode([t]) for t in row]
                    c["lens_neg"] = [self.tok.decode([t]) for t in row_neg]
        if len(self._con_cache) > 2000:
            self._con_cache.clear()
        self._con_cache[key] = out
        with self._db_lock:
            self._db.execute("INSERT OR REPLACE INTO con VALUES (?, ?)", (dkey, json.dumps(out)))
            self._db.commit()
        return out

    # ---------- logit-lens index over every neuron ----------
    def build_lens_index(self, path, k=12, chunk=1024, layers=None):
        """Top-k tokens each neuron's down column pushes up and down, for every layer. Saved as int32 arrays
        up[L, N, k], down[L, N, k]."""
        self.ensure_loaded()
        all_layers = self.adapter.layers()
        L, N = len(all_layers), self.adapter.down_proj(all_layers[0]).in_features
        up = np.zeros((L, N, k), np.int32)
        down = np.zeros((L, N, k), np.int32)
        eval_layers = range(L) if layers is None else layers
        for li in eval_layers:
            self.status = f"indexing lens {li}/{L}"
            layer = all_layers[li]
            with self.lock, torch.inference_mode():
                W = self.adapter.down_proj(layer).weight  # [hidden, inter]
                for s in range(0, N, chunk):
                    D = W[:, s:s + chunk].T[:, None]  # [c, 1, hidden]
                    logits = self.model.lm_head(self.adapter.final_norm()(D))[:, 0]
                    up[li, s:s + chunk] = logits.topk(k).indices.cpu().numpy()
                    down[li, s:s + chunk] = (-logits).topk(k).indices.cpu().numpy()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, up=up, down=down)
        self.status = "ready"
        return path
    # ---------- in-process direct-path top-k connections ----------
    def build_connections_index(self, path, k=24, layers=None):
        """Direct-path top-k upstream and downstream links for every neuron across all layer pairs."""
        self.ensure_loaded()
        all_layers = self.adapter.layers()
        L = len(all_layers)
        N = self.adapter.down_proj(all_layers[0]).in_features
        dev = self.dev
        best = {
            s: (
                torch.full((L, N, k), -1e9, device=dev),
                torch.zeros((L, N, k), dtype=torch.int16, device=dev),
                torch.zeros((L, N, k), dtype=torch.int16, device=dev),
                torch.zeros((L, N, k), dtype=torch.int8, device=dev),
            )
            for s in ("down", "up")
        }

        def merge(side, l, z, idx, other_layer, sign):
            bz, bl, bn, bs = best[side]
            cz = torch.cat([bz[l], z], 1)
            cl = torch.cat([bl[l], torch.full_like(idx, other_layer, dtype=torch.int16)], 1)
            cn = torch.cat([bn[l], idx.to(torch.int16)], 1)
            cs = torch.cat([bs[l], sign], 1)
            top = cz.topk(k, dim=1)
            bz[l], bl[l], bn[l], bs[l] = top.values, cl.gather(1, top.indices), cn.gather(1, top.indices), cs.gather(1, top.indices)

        eval_layers = list(range(L)) if layers is None else list(layers)
        down = {l: self.adapter.down_proj(all_layers[l]).weight.to(dev, torch.bfloat16) for l in eval_layers}
        gates = {l: self.adapter.gate_up_weights(all_layers[l])[0].to(dev, torch.bfloat16) for l in eval_layers}
        ups = {l: self.adapter.gate_up_weights(all_layers[l])[1].to(dev, torch.bfloat16) for l in eval_layers}
        norm = {l: self.adapter.pre_mlp_norm_scale(all_layers[l]).to(dev, torch.bfloat16) for l in eval_layers}

        with torch.inference_mode():
            for idx_j, j in enumerate(eval_layers[1:], 1):
                self.status = f"indexing connections {j}/{L}"
                for i in eval_layers[:idx_j]:
                    V = down[i] * norm[j][:, None]
                    P = (gates[j] @ V).float() * (ups[j] @ V).float()
                    M = P.abs()
                    # downstream of A (layer i): column A over B
                    zc = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-8)
                    top = zc.topk(k, dim=0)
                    sgn = torch.where(P.gather(0, top.indices) > 0, 1, -1).to(torch.int8)
                    merge("down", i, top.values.T, top.indices.T, j, sgn.T)
                    # upstream of B (layer j): row B over A
                    zr = (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-8)
                    top = zr.topk(k, dim=1)
                    sgn = torch.where(P.gather(1, top.indices) > 0, 1, -1).to(torch.int8)
                    merge("up", j, top.values, top.indices, i, sgn)
                    del P, M, zc, zr

        out = {}
        for side, (bz, bl, bn, bs) in best.items():
            out[f"{side}_z"] = bz.clamp(min=-60000).cpu().numpy().astype(np.float16)
            out[f"{side}_layer"] = bl.cpu().numpy().astype(np.int8)
            out[f"{side}_n"] = bn.cpu().numpy()
            out[f"{side}_sign"] = bs.cpu().numpy()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **out, k=k)
        self.status = "ready"
        return path

    def vocab_strings(self):
        if getattr(self, "_vocab", None) is None:
            path = os.path.join(MODEL_CACHE, "vocab.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        v = json.load(f)
                    if len(v) == self.arch["vocab_size"]:
                        self._vocab = v
                except Exception:
                    pass
            if getattr(self, "_vocab", None) is None:
                if self.tok is None:
                    self.tok = AutoTokenizer.from_pretrained(MODEL)
                n_tok = len(self.tok)
                self._vocab = [self.tok.decode([i]) for i in range(n_tok)]
                if len(self._vocab) < self.arch["vocab_size"]:
                    self._vocab.extend([f"<|extra_{i}|>" for i in range(len(self._vocab), self.arch["vocab_size"])])
                elif len(self._vocab) > self.arch["vocab_size"]:
                    self._vocab = self._vocab[:self.arch["vocab_size"]]
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump(self._vocab, f)
        return self._vocab

    # ---------- which neurons fire on a text ----------
    def all_activations(self, ids, skip_first=False):
        """Per layer: (max, argmax, min, argmin) over tokens for every neuron, on token ids [T] or [B, T]. With
        skip_first, position 0 (the attention sink) is left out and argmax/argmin count from position 1."""
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            self._capture_all, self._capture_from = {}, int(skip_first)
            try:
                x = torch.tensor(ids, device=self.dev)
                self.lm(input_ids=x if x.dim() == 2 else x[None], use_cache=False)
                return self._capture_all
            finally:
                self._capture_all, self._capture_from = None, 0

    def build_act_scale(self, path, tokens=10000, seq=512, seed=0):
        """Each neuron's largest positive and most negative activation over web text (positions >= 1)."""
        from datasets import load_dataset
        self.ensure_loaded()
        L, N = len(self.lm.layers), self.adapter.down_proj(self.lm.layers[0]).in_features
        pos, neg = np.zeros((L, N), np.float32), np.zeros((L, N), np.float32)
        buf, seen = [], 0
        for d in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed):
            buf += self.tok(d["text"], add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            while len(buf) >= seq and seen < tokens:
                chunk, buf = buf[:seq], buf[seq:]
                self.status = f"measuring neuron ranges {seen // 1000}k/{tokens // 1000}k tokens"
                acts = self.all_activations(chunk, skip_first=True)
                for l, (mx, _, mn, _) in acts.items():
                    pos[l] = np.maximum(pos[l], mx.numpy()); neg[l] = np.minimum(neg[l], mn.numpy())
                seen += seq
            if seen >= tokens:
                break
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, pos=pos, neg=neg, tokens=seen)
        self.status = "ready"
        return path

    def full_activations(self, ids):
        """Every neuron's activation at every token: float32 array [layers, tokens, inter]."""
        self.ensure_loaded()
        with self.lock, torch.inference_mode():
            self._capture_full = {}
            try:
                self.lm(input_ids=torch.tensor([ids], device=self.dev), use_cache=False)
                return torch.stack([self._capture_full[l] for l in sorted(self._capture_full)]).numpy()
            finally:
                self._capture_full = None
