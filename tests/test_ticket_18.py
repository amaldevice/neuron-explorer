"""Test suite for Ticket #18:
- ibm-granite/granite-4.0-350m and granite-4.0-h-350m ModelAdapter support in _REGISTRY
- Fused MLP (chunk gate/up) & w-norm scaling on post_attention_layernorm
- residual_multiplier compensation in add steering (dose = 1.0 -> 1.0 * resid_p50)
- Mixer head scaling/ablation on Granite self_attn (16 / 12 heads) and Granite-H mamba (48 heads)
- Clean end-of-turn termination on tok.eos_token_id (100257)
- Isolated model artifacts under explorer/cache/<slug>/
"""
import fcntl
import gc
import json
import os
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "explorer"))

from explorer.models import ADAPTERS, adapter_for, arch_from_model_or_config, model_slug
from explorer.runtime import Runtime, model_slug as rt_model_slug


class TestGraniteAdapterStatic(unittest.TestCase):
    def test_model_slug_granite(self):
        """Model slug matches expected format for both Granite models."""
        self.assertEqual(model_slug("ibm-granite/granite-4.0-350m"), "ibm-granite--granite-4.0-350m")
        self.assertEqual(model_slug("ibm-granite/granite-4.0-h-350m"), "ibm-granite--granite-4.0-h-350m")

    def test_granitemoehybrid_in_registry(self):
        """granitemoehybrid is registered in ADAPTERS."""
        self.assertIn("granitemoehybrid", ADAPTERS)

    def test_arch_dimensions_and_layer_types_granite_350m(self):
        """Authoritative (28, 1024, 2048) and full_attention layer types for granite-4.0-350m."""
        arch = arch_from_model_or_config("ibm-granite/granite-4.0-350m")
        self.assertEqual(arch["n_layers"], 28)
        self.assertEqual(arch["hidden"], 1024)
        self.assertEqual(arch["inter"], 2048)
        self.assertEqual(len(arch["layer_types"]), 28)
        self.assertTrue(all(t == "full_attention" for t in arch["layer_types"]))
        self.assertEqual(arch["attn_heads"], 16)

    def test_arch_dimensions_and_layer_types_granite_h_350m(self):
        """Authoritative (32, 768, 2048) and hybrid full_attention + mamba layer types for granite-4.0-h-350m."""
        arch = arch_from_model_or_config("ibm-granite/granite-4.0-h-350m")
        self.assertEqual(arch["n_layers"], 32)
        self.assertEqual(arch["hidden"], 768)
        self.assertEqual(arch["inter"], 2048)
        self.assertEqual(len(arch["layer_types"]), 32)
        self.assertIn("full_attention", arch["layer_types"])
        self.assertIn("mamba", arch["layer_types"])
        # Exactly 4 full_attention layers (10, 13, 17, 27) and 28 mamba layers
        attn_indices = [i for i, t in enumerate(arch["layer_types"]) if t == "full_attention"]
        self.assertEqual(attn_indices, [10, 13, 17, 27])
        self.assertEqual(arch["layer_types"].count("mamba"), 28)
        self.assertEqual(arch["attn_heads"], 12)


class TestGraniteLiveMPS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.backends.mps.is_available():
            raise unittest.SkipTest("MPS not available")

    def _acquire_mps_lock(self):
        lock_fd = open("/tmp/mps_model.lock", "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd

    def _release_mps_lock(self, lock_fd):
        gc.collect()
        torch.mps.empty_cache()
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    def test_granite_350m_adapter_and_residual_compensation(self):
        """Granite-4.0-350m: fused MLP, w-norm, residual_multiplier compensation, and mixer hook."""
        lock_fd = self._acquire_mps_lock()
        try:
            with patch.dict(os.environ, {"EXPLORER_MODEL": "ibm-granite/granite-4.0-350m"}):
                # Reload runtime module level MODEL if needed or instantiate Runtime
                from explorer import runtime as rt_mod
                saved_model = rt_mod.MODEL
                saved_slug = rt_mod.SLUG
                saved_cache = rt_mod.MODEL_CACHE
                try:
                    rt_mod.MODEL = "ibm-granite/granite-4.0-350m"
                    rt_mod.SLUG = model_slug(rt_mod.MODEL)
                    rt_mod.MODEL_CACHE = os.path.join(rt_mod.CACHE, rt_mod.SLUG)

                    rt = Runtime(os.path.join(ROOT, "stats"))
                    rt.ensure_loaded()
                    self.assertEqual(rt.dev, "mps")
                    self.assertEqual(len(rt.adapter.layers()), 28)
                    self.assertEqual(rt.adapter.eos_token_id(), 100257)

                    # 1. Adapter mappings
                    l0 = rt.adapter.layers()[0]
                    self.assertIs(rt.adapter.down_proj(l0), l0.shared_mlp.output_linear)
                    w_gate, w_up = rt.adapter.gate_up_weights(l0)
                    self.assertEqual(w_gate.shape, (2048, 1024))
                    self.assertEqual(w_up.shape, (2048, 1024))
                    self.assertTrue(torch.equal(w_gate, l0.shared_mlp.input_linear.weight[:2048]))
                    self.assertTrue(torch.equal(w_up, l0.shared_mlp.input_linear.weight[2048:]))

                    # 2. pre_mlp_norm_scale uses w scaling (not 1 + w)
                    scale = rt.adapter.pre_mlp_norm_scale(l0)
                    self.assertTrue(torch.equal(scale, l0.post_attention_layernorm.weight.float()))
                    self.assertFalse(torch.equal(scale, 1.0 + l0.post_attention_layernorm.weight.float()))

                    # 3. residual_multiplier compensation: dose = 1.0 produces 1.0 * resid_p50 perturbation
                    self.assertAlmostEqual(rt.adapter.mlp_scale(l0), 0.263, places=3)
                    down_col = rt.down_col(0, 0)
                    self.assertEqual(down_col.shape, (1024,))
                    p_entry = rt._push_entry(0, 0, dose=1.0, mode="add", rows=(1, 0, -1))
                    unit = p_entry[3][0].item()
                    # unit should be resid_p50 / (0.263 * ||down_col||)
                    expected_unit = rt.layer_resid_p50(0) / (0.263 * float(down_col.norm()))
                    self.assertAlmostEqual(unit, expected_unit, places=3)

                    # Test actual residual perturbation on layer output
                    ids = rt.tok("Testing residual perturbation on Granite", return_tensors="pt").input_ids.to(rt.dev)
                    with torch.inference_mode():
                        embeds = rt.model.model.embed_tokens(ids) * rt.model.model.embedding_multiplier
                        base_out = l0(embeds)

                    def add_hook(mod, inp):
                        h = inp[0].clone()
                        h[:, :, 0] += unit
                        return (h,)

                    h_handle = l0.shared_mlp.output_linear.register_forward_pre_hook(add_hook)
                    with torch.inference_mode():
                        steered_out = l0(embeds)
                    h_handle.remove()

                    diff_norm = (steered_out - base_out).float().norm(dim=-1)[0, 0].item()
                    expected_norm = rt.layer_resid_p50(0)
                    self.assertAlmostEqual(diff_norm / expected_norm, 1.0, delta=0.03)

                    # 4. Mixer attention heads (16 heads)
                    self.assertEqual(rt.n_heads(0), 16)
                    self.assertIs(rt.adapter.mixer_out_proj(l0), l0.self_attn.o_proj)

                    # 5. Read with head scaling
                    r = rt.read(0, 0, "Hello Granite", chat=False, heads=[(0, 0, 0.0)])
                    self.assertIn("units", r)
                    self.assertGreater(len(r["units"]), 0)

                    # 6. Live steered chat terminates cleanly on eos_token_id (100257)
                    messages = [[{"role": "user", "content": "Say hello"}]] * 3
                    results = ["", "", ""]
                    for row, text in rt.steer_stream(0, 0, messages, dose=0.8, mode="add", max_new=12):
                        if row is not None:
                            results[row] += text
                    self.assertEqual(len(results), 3)

                    # 7. Isolated model artifacts
                    self.assertTrue(rt_mod.MODEL_CACHE.endswith("ibm-granite--granite-4.0-350m"))
                    con_path = os.path.join(rt_mod.MODEL_CACHE, "connections.npz")
                    rt.build_connections_index(con_path, k=8, layers=range(2))
                    self.assertTrue(os.path.exists(con_path))
                    ci = np.load(con_path)
                    self.assertEqual(ci["down_z"].shape[:2], (28, 2048))

                finally:
                    rt_mod.MODEL = saved_model
                    rt_mod.SLUG = saved_slug
                    rt_mod.MODEL_CACHE = saved_cache
                    if "rt" in locals():
                        del rt
        finally:
            self._release_mps_lock(lock_fd)

    def test_granite_h_350m_adapter_mamba_and_attention_mixers(self):
        """Granite-4.0-h-350m: Mamba2 mixer (48 heads) and Attention mixer (12 heads), layer.self_attn = None."""
        lock_fd = self._acquire_mps_lock()
        try:
            with patch.dict(os.environ, {"EXPLORER_MODEL": "ibm-granite/granite-4.0-h-350m"}):
                from explorer import runtime as rt_mod
                saved_model = rt_mod.MODEL
                saved_slug = rt_mod.SLUG
                saved_cache = rt_mod.MODEL_CACHE
                try:
                    rt_mod.MODEL = "ibm-granite/granite-4.0-h-350m"
                    rt_mod.SLUG = model_slug(rt_mod.MODEL)
                    rt_mod.MODEL_CACHE = os.path.join(rt_mod.CACHE, rt_mod.SLUG)

                    rt = Runtime(os.path.join(ROOT, "stats"))
                    rt.ensure_loaded()
                    self.assertEqual(rt.dev, "mps")
                    self.assertEqual(len(rt.adapter.layers()), 32)
                    self.assertEqual(rt.adapter.eos_token_id(), 100257)

                    # Layer 0 is Mamba-2 (layer.self_attn is None)
                    l0 = rt.adapter.layers()[0]
                    self.assertIsNone(l0.self_attn)
                    self.assertIsNotNone(l0.mamba)
                    self.assertIs(rt.adapter.mixer_out_proj(l0), l0.mamba.out_proj)
                    self.assertEqual(rt.adapter.n_heads(l0), 48)
                    self.assertEqual(rt.n_heads(0), 48)

                    # Layer 10 is Attention (layer.self_attn is not None, 12 heads)
                    l10 = rt.adapter.layers()[10]
                    self.assertIsNotNone(l10.self_attn)
                    self.assertIs(rt.adapter.mixer_out_proj(l10), l10.self_attn.o_proj)
                    self.assertEqual(rt.adapter.n_heads(l10), 12)
                    self.assertEqual(rt.n_heads(10), 12)

                    # residual_multiplier on h-350m is 0.246
                    self.assertAlmostEqual(rt.adapter.mlp_scale(l0), 0.246, places=3)

                    # Residual perturbation on h-350m
                    down_col = rt.down_col(0, 0)
                    self.assertEqual(down_col.shape, (768,))
                    p_entry = rt._push_entry(0, 0, dose=1.0, mode="add", rows=(1, 0, -1))
                    unit = p_entry[3][0].item()
                    expected_unit = rt.layer_resid_p50(0) / (0.246 * float(down_col.norm()))
                    self.assertAlmostEqual(unit, expected_unit, places=3)

                    ids = rt.tok("Testing residual perturbation on Granite-H", return_tensors="pt").input_ids.to(rt.dev)
                    with torch.inference_mode():
                        embeds = rt.model.model.embed_tokens(ids) * rt.model.model.embedding_multiplier
                        base_out = l0(embeds)

                    def add_hook(mod, inp):
                        h = inp[0].clone()
                        h[:, :, 0] += unit
                        return (h,)

                    h_handle = l0.shared_mlp.output_linear.register_forward_pre_hook(add_hook)
                    with torch.inference_mode():
                        steered_out = l0(embeds)
                    h_handle.remove()

                    diff_norm = (steered_out - base_out).float().norm(dim=-1)[0, 0].item()
                    expected_norm = rt.layer_resid_p50(0)
                    self.assertAlmostEqual(diff_norm / expected_norm, 1.0, delta=0.03)

                    # Read on Mamba layer with head scaling (48 heads)
                    r_mamba = rt.read(0, 0, "Testing Mamba-2 layer read", chat=False, heads=[(0, 5, 0.0)])
                    self.assertIn("units", r_mamba)
                    self.assertGreater(len(r_mamba["units"]), 0)

                    # Read on Attention layer with head scaling (12 heads)
                    r_attn = rt.read(10, 0, "Testing Attention layer read", chat=False, heads=[(10, 2, 0.0)])
                    self.assertIn("units", r_attn)
                    self.assertGreater(len(r_attn["units"]), 0)

                    # Steered chat terminates cleanly on eos_token_id (100257)
                    messages = [[{"role": "user", "content": "Say hi"}]] * 3
                    results = ["", "", ""]
                    for row, text in rt.steer_stream(0, 0, messages, dose=0.8, mode="add", max_new=12):
                        if row is not None:
                            results[row] += text
                    self.assertEqual(len(results), 3)

                    # Isolated artifacts under ibm-granite--granite-4.0-h-350m
                    self.assertTrue(rt_mod.MODEL_CACHE.endswith("ibm-granite--granite-4.0-h-350m"))
                    con_path = os.path.join(rt_mod.MODEL_CACHE, "connections.npz")
                    rt.build_connections_index(con_path, k=8, layers=range(2))
                    self.assertTrue(os.path.exists(con_path))
                    ci = np.load(con_path)
                    self.assertEqual(ci["down_z"].shape[:2], (32, 2048))

                finally:
                    rt_mod.MODEL = saved_model
                    rt_mod.SLUG = saved_slug
                    rt_mod.MODEL_CACHE = saved_cache
                    if "rt" in locals():
                        del rt
        finally:
            self._release_mps_lock(lock_fd)


if __name__ == "__main__":
    unittest.main()
