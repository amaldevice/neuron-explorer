"""Subject-model adapters: where a model family keeps the pieces the explorer reads and steers.

The registry is keyed by `config.model_type`; an unknown type fails fast with the supported keys. The model is loaded
with AutoModelForCausalLM, which for Qwen3.5 skips the vision tower and puts the text backbone at `m.model`.
"""
import json
import os
import re

import torch

MODEL = os.environ.get("EXPLORER_MODEL", "LiquidAI/LFM2.5-350M")


def model_slug(model_id):
    """Per-model cache directory name: "Qwen/Qwen3.5-0.8B" -> "qwen--qwen3.5-0.8b"."""
    return model_id.lower().replace("/", "--")


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_id):
    """The text model in bfloat16 on MPS / CUDA / CPU, eval mode."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device()).eval()


class ModelAdapter:
    """Qwen3.5: layers at m.model.layers, MLP at layer.mlp (gate_proj / up_proj / down_proj), zero-centred pre-MLP norm
    (1 + w) at post_attention_layernorm, final norm at m.model.norm, token mixer out projection at self_attn.o_proj
    (full attention) or linear_attn.out_proj (linear attention)."""

    def __init__(self, model, tok=None):
        self.model, self.tok = model, tok
        self.text_model = getattr(model.model, "language_model", model.model)

    def layers(self):
        return self.text_model.layers

    def down_proj(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return layer.mlp.down_proj

    def gate_up_weights(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight

    def pre_mlp_norm_scale(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return 1.0 + layer.post_attention_layernorm.weight.float()

    def mlp_scale(self, layer):
        """Factor the MLP output is multiplied by before joining the residual stream (1 unless the family has one)."""
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return float(getattr(layer, "residual_multiplier", 1.0))

    def final_norm(self):
        return self.text_model.norm

    def mixer_out_proj(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        a = getattr(layer, "self_attn", None)
        return a.o_proj if a is not None else layer.linear_attn.out_proj

    def n_heads(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        a = getattr(layer, "self_attn", None)
        return a.o_proj.in_features // a.head_dim if a is not None else layer.linear_attn.num_v_heads

    def mixer(self, layer):
        return self.mixer_out_proj(layer), self.n_heads(layer)

    def eos_token_id(self):
        return self.tok.eos_token_id
    def hook_mlp(self, layer, s):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        layer.mlp.gate_proj.register_forward_hook(lambda m, i, o: setattr(s, "gate", o))
        layer.mlp.up_proj.register_forward_hook(lambda m, i, o: setattr(s, "up", o))
        layer.mlp.down_proj.register_forward_pre_hook(lambda m, i: s.update(i[0]))


class Lfm2Adapter(ModelAdapter):
    """LiquidAI LFM2: layers at m.model.layers, MLP at layer.feed_forward (w1/w3/w2),
    pre-MLP norm at layer.ffn_norm (w scaling), final norm at m.model.embedding_norm,
    mixer out projection at self_attn.out_proj (full_attention) or conv.out_proj (conv)."""

    def down_proj(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return layer.feed_forward.w2

    def gate_up_weights(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return layer.feed_forward.w1.weight, layer.feed_forward.w3.weight

    def pre_mlp_norm_scale(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return layer.ffn_norm.weight.float()

    def final_norm(self):
        return self.text_model.embedding_norm

    def mixer_out_proj(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        if getattr(layer, "self_attn", None) is not None:
            return layer.self_attn.out_proj
        return layer.conv.out_proj

    def n_heads(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        a = getattr(layer, "self_attn", None)
        if a is not None:
            return a.out_proj.in_features // a.head_dim
        return 0

    def hook_mlp(self, layer, s):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        layer.feed_forward.w1.register_forward_hook(lambda m, i, o: setattr(s, "gate", o))
        layer.feed_forward.w3.register_forward_hook(lambda m, i, o: setattr(s, "up", o))
        layer.feed_forward.w2.register_forward_pre_hook(lambda m, i: s.update(i[0]))


class GraniteMoeHybridAdapter(ModelAdapter):
    """Granite-4.0 / Granite-4.0-H: layers at m.model.layers, MLP at layer.shared_mlp (input_linear / output_linear),
    standard pre-MLP norm (w) at post_attention_layernorm, final norm at m.model.norm, token mixer at self_attn.o_proj
    or mamba.out_proj."""

    def down_proj(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return layer.shared_mlp.output_linear

    def gate_up_weights(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        gate, up = layer.shared_mlp.input_linear.weight.chunk(2, dim=0)
        return gate, up

    def pre_mlp_norm_scale(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return layer.post_attention_layernorm.weight.float()

    def mlp_scale(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        return float(getattr(layer, "residual_multiplier", 1.0))

    def final_norm(self):
        return self.text_model.norm

    def mixer_out_proj(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        a = getattr(layer, "self_attn", None)
        if a is not None:
            return a.o_proj
        m = getattr(layer, "mamba", None)
        if m is not None:
            return m.out_proj
        raise ValueError(f"layer {layer} has neither self_attn nor mamba mixer")

    def n_heads(self, layer):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        a = getattr(layer, "self_attn", None)
        if a is not None:
            return a.o_proj.in_features // a.head_dim
        m = getattr(layer, "mamba", None)
        if m is not None:
            return m.num_heads
        raise ValueError(f"layer {layer} has neither self_attn nor mamba mixer")

    def hook_mlp(self, layer, s):
        if isinstance(layer, int):
            layer = self.layers()[layer]
        def hook_in(m, i, o):
            g, u = o.chunk(2, dim=-1)
            s.gate, s.up = g, u
        layer.shared_mlp.input_linear.register_forward_hook(hook_in)
        layer.shared_mlp.output_linear.register_forward_pre_hook(lambda m, i: s.update(i[0]))


ADAPTERS = {
    "qwen3_5": ModelAdapter,
    "qwen3_5_text": ModelAdapter,
    "lfm2": Lfm2Adapter,
    "granitemoehybrid": GraniteMoeHybridAdapter,
}
_REGISTRY = ADAPTERS
ADAPTERS_MAP = ADAPTERS
ADAPTER_REGISTRY = ADAPTERS


def adapter_for(model, tok=None):
    mt = getattr(getattr(model, "config", None), "model_type", None)
    if mt not in ADAPTERS:
        raise ValueError(f"unsupported model_type {mt!r}; supported: {sorted(ADAPTERS)}")
    return ADAPTERS[mt](model, tok)


_DOWN = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.(?:mlp\.down_proj|feed_forward\.w2|shared_mlp\.output_linear)\.weight$"
)


def _local_safetensors(model_id):
    """Paths of the model's cached .safetensors files, or []."""
    from huggingface_hub import try_to_load_from_cache
    idx = try_to_load_from_cache(model_id, "model.safetensors.index.json")
    if isinstance(idx, str):
        with open(idx) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        paths = [try_to_load_from_cache(model_id, s) for s in shards]
    else:
        paths = [try_to_load_from_cache(model_id, "model.safetensors")]
    return [p for p in paths if isinstance(p, str)]


def arch_from_model_or_config(model_id):
    """(n_layers, hidden, inter) read from the local safetensors headers (down_proj is [hidden, inter]), and the
    config's when the weights are not on disk, plus the config's head counts and layer types."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id).get_text_config()
    inter = getattr(cfg, "shared_intermediate_size", getattr(cfg, "intermediate_size", None))
    if getattr(cfg, "block_auto_adjust_ff_dim", False):
        inter = int(2 * inter / 3)
        mult = getattr(cfg, "block_ffn_dim_multiplier", None)
        if mult is not None:
            inter = int(mult * inter)
        b_mult = getattr(cfg, "block_multiple_of", 256)
        inter = b_mult * ((inter + b_mult - 1) // b_mult)
    ltypes = list(getattr(cfg, "layer_types", []))
    if getattr(cfg, "model_type", None) == "granitemoehybrid":
        ltypes = [
            "mamba" if lt in ("linear_attention", "mamba") else ("full_attention" if lt == "attention" else lt)
            for lt in ltypes
        ]
    arch = {
        "n_layers": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size,
        "inter": inter,
        "vocab_size": cfg.vocab_size,
        "layer_types": ltypes,
        "attn_heads": getattr(cfg, "num_attention_heads", None),
        "kv_heads": getattr(cfg, "num_key_value_heads", None),
        "linear_k_heads": getattr(cfg, "linear_num_key_heads", None),
        "linear_v_heads": getattr(cfg, "linear_num_value_heads", None),
    }
    from safetensors import safe_open
    layers, shape = set(), None
    for path in _local_safetensors(model_id):
        with safe_open(path, "pt") as f:
            for k in f.keys():
                if m := _DOWN.match(k):
                    layers.add(int(m.group(1)))
                    shape = f.get_slice(k).get_shape()
    if layers and shape:
        arch.update(n_layers=len(layers), hidden=shape[0], inter=shape[1])
    return arch
