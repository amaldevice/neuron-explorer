"""Test suite for Ticket #17:
LiquidAI/LFM2.5-350M Subject model adapter, conv Mixer support, and dynamic grid layout.
"""
import fcntl
import gc
import json
import math
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from transformers import AutoConfig, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "explorer"))

from explorer.models import (
    ADAPTERS,
    _REGISTRY,
    adapter_for,
    arch_from_model_or_config,
    model_slug,
    MODEL as DEFAULT_EXPLORER_MODEL,
)
from explorer.runtime import MODEL, MODEL_CACHE, Runtime
from explorer.server import ARCH, PORT, app, rt


class TestLfm2AdapterRegistrationAndArch(unittest.TestCase):
    def test_default_model(self):
        """Default Subject model is LiquidAI/LFM2.5-350M."""
        self.assertEqual(DEFAULT_EXPLORER_MODEL, "LiquidAI/LFM2.5-350M")
        self.assertEqual(MODEL, "LiquidAI/LFM2.5-350M")
        self.assertEqual(model_slug("LiquidAI/LFM2.5-350M"), "liquidai--lfm2.5-350m")

    def test_registry(self):
        """lfm2 is registered in _REGISTRY and ADAPTERS."""
        self.assertIn("lfm2", ADAPTERS)
        self.assertIn("lfm2", _REGISTRY)
        self.assertIs(ADAPTERS["lfm2"], _REGISTRY["lfm2"])

    def test_arch_authoritative_inter(self):
        """Authoritative inter=4608 from safetensors header and config fallback."""
        arch = arch_from_model_or_config("LiquidAI/LFM2.5-350M")
        self.assertEqual(arch["n_layers"], 16)
        self.assertEqual(arch["hidden"], 1024)
        self.assertEqual(arch["inter"], 4608)
        self.assertIn("conv", arch["layer_types"])
        self.assertIn("full_attention", arch["layer_types"])
        self.assertEqual(len(arch["layer_types"]), 16)
        self.assertEqual(arch["layer_types"].count("conv"), 10)
        self.assertEqual(arch["layer_types"].count("full_attention"), 6)

    def test_config_fallback_adjust_ff_dim(self):
        """Fallback without safetensors applies block_auto_adjust_ff_dim to intermediate_size=6656 -> 4608."""
        cfg = AutoConfig.from_pretrained("LiquidAI/LFM2.5-350M").get_text_config()
        self.assertEqual(cfg.intermediate_size, 6656)
        self.assertTrue(getattr(cfg, "block_auto_adjust_ff_dim", False))
        with patch("explorer.models._local_safetensors", return_value=[]):
            arch = arch_from_model_or_config("LiquidAI/LFM2.5-350M")
            self.assertEqual(arch["inter"], 4608)


class TestOfflineScripts(unittest.TestCase):
    def test_build_layout_single_safetensors_and_dynamic_grid(self):
        """build_layout.py runs cleanly against LiquidAI/LFM2.5-350M with 68x68 grid."""
        out_path = "/tmp/test_layout_writes.npz"
        if os.path.exists(out_path):
            os.remove(out_path)
        cmd = [
            sys.executable,
            os.path.join(ROOT, "explorer", "build_layout.py"),
            "--model", "LiquidAI/LFM2.5-350M",
            "--kind", "writes",
            "--layers", "1",
            "--out", out_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"build_layout failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        self.assertTrue(os.path.exists(out_path))
        data = np.load(out_path)
        cells = data["cell"]
        self.assertEqual(cells.shape[0], 1)
        self.assertEqual(cells.shape[1], 4608)
        side = math.ceil(math.sqrt(4608))
        self.assertEqual(side, 68)
        self.assertLess(cells.max(), side * side)
        self.assertGreaterEqual(cells.min(), 0)
        self.assertEqual(len(np.unique(cells[0])), 4608)
        os.remove(out_path)

    def test_build_connections_single_safetensors_and_w_norm(self):
        """build_connections.py runs cleanly against LiquidAI/LFM2.5-350M using w norm scaling."""
        out_path = "/tmp/test_connections.npz"
        if os.path.exists(out_path):
            os.remove(out_path)
        cmd = [
            sys.executable,
            os.path.join(ROOT, "explorer", "build_connections.py"),
            "--model", "LiquidAI/LFM2.5-350M",
            "--k", "12",
            "--layers", "2",
            "--out", out_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"build_connections failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        self.assertTrue(os.path.exists(out_path))
        data = np.load(out_path)
        self.assertEqual(data["k"], 12)
        self.assertEqual(data["down_z"].shape[:2], (2, 4608))
        self.assertEqual(data["up_z"].shape[:2], (2, 4608))
        os.remove(out_path)


class TestConvMixerAndHeadAblation(unittest.TestCase):
    def test_head_rejection_and_conv_ablation(self):
        """read and next_logprobs support head='all' on conv, reject integer head indices, per-head on full_attention."""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")

        lock_path = "/tmp/mps_model.lock"
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                test_rt = Runtime(os.path.join(ROOT, "stats"))
                test_rt.ensure_loaded()
                self.assertEqual(test_rt.n_heads(0), 0)   # L0 is conv
                self.assertEqual(test_rt.n_heads(2), 16)  # L2 is full_attention

                # Conv layer integer head index rejected with exact error message
                with self.assertRaises(ValueError) as ctx:
                    test_rt.read(0, 10, "Hello world", chat=False, heads=[(0, 0, 0.0)])
                self.assertIn("layer 0 has 0 heads (conv)", str(ctx.exception))

                with self.assertRaises(ValueError) as ctx:
                    test_rt.next_logprobs("Hello world", chat=False, heads=[(0, 1, 0.0)])
                self.assertIn("layer 0 has 0 heads (conv)", str(ctx.exception))

                # Whole-mixer ablation on conv layer (head=None / head="all")
                r_base = test_rt.read(0, 10, "Hello world", chat=False)
                r_ablated = test_rt.read(0, 10, "Hello world", chat=False, heads=[(0, "all", 0.0)])
                self.assertIn("units", r_base)
                self.assertIn("units", r_ablated)

                base_lp, ablated_lp = test_rt.next_logprobs("Hello world", chat=False, heads=[(0, None, 0.0)])
                self.assertIsNotNone(ablated_lp)
                self.assertFalse(torch.allclose(base_lp, ablated_lp))

                # Full attention layer per-head scaling
                r_head = test_rt.read(3, 10, "Hello world", chat=False, heads=[(2, 0, 0.0)])
                self.assertIn("units", r_head)

                # layer_resid_p50 cleanly falls back on REFERENCE_TEXT when stats file is missing
                p50 = test_rt.layer_resid_p50(0)
                self.assertIsInstance(p50, float)
                self.assertGreater(p50, 0.0)
            finally:
                if "test_rt" in locals():
                    del test_rt
                gc.collect()
                torch.mps.empty_cache()
                fcntl.flock(lock_fd, fcntl.LOCK_UN)


class TestLiveMpsBringupLfm2(unittest.TestCase):
    def test_live_steer_chat_and_fires_on_mps(self):
        """End-to-end MPS bring-up under hardware lock: 3-column steer stream and fires endpoint."""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS not available")

        lock_path = "/tmp/mps_model.lock"
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                test_rt = Runtime(os.path.join(ROOT, "stats"))
                test_rt.ensure_loaded()
                self.assertEqual(len(test_rt.adapter.layers()), 16)
                self.assertEqual(test_rt.adapter.eos_token_id(), 7)

                # 3-column steered chat generation: 8 tokens
                messages = [[{"role": "user", "content": "Explain gravity in one sentence."}]] * 3
                results_add = ["", "", ""]
                for row, text in test_rt.steer_stream(0, 20, messages, dose=0.8, mode="add", max_new=8):
                    if row is not None:
                        results_add[row] += text
                self.assertEqual(len(results_add), 3)

                results_mul = ["", "", ""]
                for row, text in test_rt.steer_stream(0, 20, messages, dose=2.0, mode="mul", max_new=8):
                    if row is not None:
                        results_mul[row] += text
                self.assertEqual(len(results_mul), 3)

                # Test connections index build and matching top-24 links
                con_path = os.path.join(MODEL_CACHE, "connections.npz")
                test_rt.build_connections_index(con_path, k=24, layers=range(2))
                self.assertTrue(os.path.exists(con_path))
                ci = test_rt.con_index()
                self.assertIsNotNone(ci)
                self.assertEqual(ci["down_z"].shape[:2], (16, 4608))
                self.assertEqual(ci["k"], 24)

                con_match = test_rt.connections(0, 10, k=24)
                self.assertIn("down", con_match)
                self.assertIn("up", con_match)
                # Verify Runtime.connections matches connections.npz top-24 links
                for i, link in enumerate(con_match["down"]):
                    self.assertEqual(link["layer"], int(ci["down_layer"][0, 10, i]))
                    self.assertEqual(link["n"], int(ci["down_n"][0, 10, i]))

                # Preload artifacts: vocab.json, lens_index.npz, act_scale.npz
                vocab = test_rt.vocab_strings()
                self.assertEqual(len(vocab), 65536)
                self.assertTrue(os.path.exists(os.path.join(MODEL_CACHE, "vocab.json")))

                lens_path = os.path.join(MODEL_CACHE, "lens_index.npz")
                test_rt.build_lens_index(lens_path, k=6, chunk=512, layers=range(1))
                self.assertTrue(os.path.exists(lens_path))

                # Create synthetic act_scale if not existing so fires can run
                act_scale_path = os.path.join(MODEL_CACHE, "act_scale.npz")
                if not os.path.exists(act_scale_path):
                    pos = np.ones((16, 4608), dtype=np.float32)
                    neg = -np.ones((16, 4608), dtype=np.float32)
                    np.savez(act_scale_path, pos=pos, neg=neg, tokens=1000)

                # Test fires endpoint via TestClient
                with patch("explorer.server.rt", test_rt):
                    client = TestClient(app)
                    resp = client.get("/api/fires?text=Hello+world&limit=5")
                    self.assertEqual(resp.status_code, 200)
                    data = resp.json()
                    self.assertTrue(data.get("ready"))
                    self.assertIn("results", data)
                    self.assertGreater(len(data["results"]), 0)

                    # Test API rejected integer head on conv layer
                    resp_act = client.post("/api/agent/activations", json={
                        "layer": 0, "n": 10, "text": "Hello", "heads": [{"layer": 0, "head": 0, "dose": 0}]
                    })
                    self.assertEqual(resp_act.status_code, 200)
                    self.assertIn("error", resp_act.json())
                    self.assertIn("layer 0 has 0 heads (conv)", resp_act.json()["error"])
            finally:
                if "test_rt" in locals():
                    del test_rt
                gc.collect()
                torch.mps.empty_cache()
                fcntl.flock(lock_fd, fcntl.LOCK_UN)


if __name__ == "__main__":
    unittest.main()
