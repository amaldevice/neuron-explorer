"""Subject-model adapters: where a model family keeps the pieces the explorer reads and steers.

The registry is keyed by `config.model_type`; an unknown type fails fast with the supported keys. The model is loaded
with AutoModelForCausalLM, which for Qwen3.5 skips the vision tower and puts the text backbone at `m.model`.
"""
import json
import os
import re

import torch

MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3.5-0.8B")


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
        return layer.mlp.down_proj

    def gate_up_weights(self, layer):
        return layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight

    def pre_mlp_norm_scale(self, layer):
        return 1.0 + layer.post_attention_layernorm.weight.float()

    def mlp_scale(self, layer):
        """Factor the MLP output is multiplied by before joining the residual stream (1 unless the family has one)."""
        return float(getattr(layer, "residual_multiplier", 1.0))

    def final_norm(self):
        return self.text_model.norm

    def mixer_out_proj(self, layer):
        a = getattr(layer, "self_attn", None)
        return a.o_proj if a is not None else layer.linear_attn.out_proj

    def n_heads(self, layer):
        a = getattr(layer, "self_attn", None)
        return a.o_proj.in_features // a.head_dim if a is not None else layer.linear_attn.num_v_heads

    def eos_token_id(self):
        return self.tok.eos_token_id


ADAPTERS = {"qwen3_5": ModelAdapter, "qwen3_5_text": ModelAdapter}


def adapter_for(model, tok=None):
    mt = getattr(getattr(model, "config", None), "model_type", None)
    if mt not in ADAPTERS:
        raise ValueError(f"unsupported model_type {mt!r}; supported: {sorted(ADAPTERS)}")
    return ADAPTERS[mt](model, tok)


_DOWN = re.compile(r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.down_proj\.weight$")


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
    arch = {"n_layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size, "inter": cfg.intermediate_size,
            "vocab_size": cfg.vocab_size, "layer_types": list(cfg.layer_types), "attn_heads": cfg.num_attention_heads,
            "kv_heads": cfg.num_key_value_heads, "linear_k_heads": cfg.linear_num_key_heads,
            "linear_v_heads": cfg.linear_num_value_heads}
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
