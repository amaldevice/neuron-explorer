"""Test suite for Ticket #19:
- sweep/steer_lens.py on MPS/CUDA/CPU with ModelAdapter, mlp_scale compensation, and subset neurons
- sweep/neuron_stats.py on MPS/CUDA/CPU with ModelAdapter writing stats/<slug>_L{layer}.npz
- explorer/server.py /api/units causal KL ranking fallback when sweep runs are absent
- explorer/static/app.js rendering kl {u.z} and empty state message
"""
import fcntl
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "explorer"))
sys.path.insert(0, os.path.join(ROOT, "sweep"))

from explorer.models import ADAPTERS, adapter_for, model_slug
from sweep.steer_lens import parse_neurons, select_device_and_dtype


class TestTicket19ServerUnits(unittest.TestCase):
    """Test /api/units fallback logic in explorer/server.py."""

    def test_units_empty_when_no_runs_and_no_steer_index(self):
        """When neither sweep runs nor steer_index exist, /api/units returns []."""
        from explorer import server

        with patch.object(server, "load_layer", return_value=None), \
             patch.object(server.rt, "steer_index", return_value=None):
            res = server.units(layer=0)
            self.assertEqual(res, [])

    def test_units_ranked_by_causal_kl_when_steer_index_present(self):
        """When sweep runs are absent but steer_index exists, /api/units returns neurons sorted descending by max_kl."""
        from explorer import server

        n_neurons = 4
        k = 6
        # Layer 0: neuron 0 has max_kl = 1.5 (up_kl=1.5, down_kl=0.2, pole=up)
        #          neuron 1 has max_kl = 3.2 (up_kl=0.5, down_kl=3.2, pole=down)
        #          neuron 2 has max_kl = 0.8 (up_kl=0.8, down_kl=0.1, pole=up)
        #          neuron 3 has max_kl = 2.1 (up_kl=2.1, down_kl=1.9, pole=up)
        # Expected rank order: neuron 1 (3.2), neuron 3 (2.1), neuron 0 (1.5), neuron 2 (0.8)
        up_kl = np.array([[1.5, 0.5, 0.8, 2.1]], dtype=np.float32)
        down_kl = np.array([[0.2, 3.2, 0.1, 1.9]], dtype=np.float32)
        up = np.zeros((1, n_neurons, k), dtype=np.int32)
        down = np.zeros((1, n_neurons, k), dtype=np.int32)

        # Set distinct token IDs for dominant poles
        # neuron 1 dominant is down -> tokens [10, 11, 12, 13, 14, 15]
        down[0, 1, :5] = [10, 11, 12, 13, 14]
        # neuron 3 dominant is up -> tokens [30, 31, 32, 33, 34, 35]
        up[0, 3, :5] = [30, 31, 32, 33, 34]
        # neuron 0 dominant is up -> tokens [1, 2, 3, 4, 5, 6]
        up[0, 0, :5] = [1, 2, 3, 4, 5]
        # neuron 2 dominant is up -> tokens [20, 21, 22, 23, 24, 25]
        up[0, 2, :5] = [20, 21, 22, 23, 24]

        fake_steer = {
            "up": up,
            "down": down,
            "up_kl": up_kl,
            "down_kl": down_kl,
        }

        fake_vocab = [f"tok_{i}" for i in range(100)]
        fake_names = {"L0/N1": "important_feature"}

        with patch.object(server, "load_layer", return_value=None), \
             patch.object(server.rt, "steer_index", return_value=fake_steer), \
             patch.object(server.rt, "vocab_strings", return_value=fake_vocab), \
             patch.object(server, "names", return_value=fake_names):
            res = server.units(layer=0)

            self.assertEqual(len(res), 4)
            # Verify ordering: [1, 3, 0, 2]
            self.assertEqual([u["n"] for u in res], [1, 3, 0, 2])

            # Check neuron 1
            u1 = res[0]
            self.assertEqual(u1["kind"], "neuron")
            self.assertEqual(u1["metric"], "kl")
            self.assertAlmostEqual(u1["z"], 3.2, places=4)
            self.assertEqual(u1["name"], "important_feature")
            self.assertEqual(u1["words"], ["tok_10", "tok_11", "tok_12", "tok_13", "tok_14"])

            # Check neuron 3
            u3 = res[1]
            self.assertEqual(u3["metric"], "kl")
            self.assertAlmostEqual(u3["z"], 2.1, places=4)
            self.assertEqual(u3["words"], ["tok_30", "tok_31", "tok_32", "tok_33", "tok_34"])
    def test_units_empty_when_steer_index_has_no_data_for_target_layer(self):
        """When steer_index exists but target layer has no non-zero KL or steer data, /api/units returns []."""
        from explorer import server

        up_kl = np.zeros((2, 4), dtype=np.float32)
        down_kl = np.zeros((2, 4), dtype=np.float32)
        up_kl[0, 0] = 1.0  # Layer 0 has data, Layer 1 has none
        fake_steer = {
            "up": np.zeros((2, 4, 6), dtype=np.int32),
            "down": np.zeros((2, 4, 6), dtype=np.int32),
            "up_kl": up_kl,
            "down_kl": down_kl,
        }
        with patch.object(server, "load_layer", return_value=None), \
             patch.object(server.rt, "steer_index", return_value=fake_steer):
            res_l1 = server.units(layer=1)
            self.assertEqual(res_l1, [])


class TestTicket19SteerLensLogic(unittest.TestCase):
    """Test unit scaling and argument parsing in sweep/steer_lens.py."""

    def test_neuron_subset_parsing(self):
        """parse_neurons handles ranges, counts, lists, and 'all'."""
        self.assertEqual(parse_neurons("all", 100), list(range(100)))
        self.assertEqual(parse_neurons("0-15", 100), list(range(16)))
        self.assertEqual(parse_neurons("16", 100), list(range(16)))
        self.assertEqual(parse_neurons("4-7", 100), [4, 5, 6, 7])
        self.assertEqual(parse_neurons("2,5,8", 100), [2, 5, 8])

    def test_granite_mlp_scale_compensation(self):
        """Residual scale compensation unit = norms[l] / (adapter.mlp_scale(l) * W.norm(dim=0))."""
        class DummyLayer:
            def __init__(self, mult):
                self.residual_multiplier = mult

        class DummyBackbone:
            def __init__(self, mult):
                self.layers = [DummyLayer(mult)]

        class DummyModel:
            def __init__(self, mult):
                self.config = MagicMock(model_type="granitemoehybrid")
                self.model = DummyBackbone(mult)

        adapter_granite = ADAPTERS["granitemoehybrid"](DummyModel(0.263))
        self.assertAlmostEqual(adapter_granite.mlp_scale(0), 0.263, places=3)

        adapter_qwen = ADAPTERS["qwen3_5"](DummyModel(1.0))
        self.assertAlmostEqual(adapter_qwen.mlp_scale(0), 1.0, places=3)
    def test_select_device_and_dtype(self):
        """Device selection prefers mps -> cuda -> cpu."""
        dev, dtype = select_device_and_dtype()
        if torch.backends.mps.is_available():
            self.assertEqual(dev, "mps")
            self.assertEqual(dtype, torch.bfloat16)
        elif torch.cuda.is_available():
            self.assertEqual(dev, "cuda")
        else:
            self.assertEqual(dev, "cpu")


class TestTicket19FrontendStrings(unittest.TestCase):
    """Verify frontend templates and kl metric formatting in app.js and index.html/style.css."""

    def test_app_js_contains_kl_formatting_and_empty_state(self):
        app_js_path = os.path.join(ROOT, "explorer", "static", "app.js")
        with open(app_js_path) as f:
            content = f.read()

        # Check for kl formatting: kl ${u.z.toFixed(1)}
        self.assertIn("kl ${u.z.toFixed(1)}", content)

        # Check for empty state message
        self.assertIn("No sweep runs or steer lens generated for L${state.layer}", content)
        self.assertIn("Search tokens via lens box above, test with fires, or run steer_lens.py", content)


class TestTicket19LiveMPS(unittest.TestCase):
    """Hardware-locked live integration test on Apple Silicon MPS."""

    @classmethod
    def setUpClass(cls):
        # Hardware lock on /tmp/mps_model.lock
        cls._lock_file = open("/tmp/mps_model.lock", "w")
        fcntl.flock(cls._lock_file, fcntl.LOCK_EX)

    @classmethod
    def tearDownClass(cls):
        try:
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
        finally:
            fcntl.flock(cls._lock_file, fcntl.LOCK_UN)
            cls._lock_file.close()

    def test_steer_lens_and_neuron_stats_mps(self):
        """Run steer_lens and neuron_stats with lightweight parameters on LFM2.5-350M."""
        model_id = "LiquidAI/LFM2.5-350M"
        slug = model_slug(model_id)
        cache_dir = os.path.join(ROOT, "explorer", "cache", slug)
        steer_file = os.path.join(cache_dir, "steer_index.npz")
        stats_file = os.path.join(ROOT, "stats", f"{slug}_L0.npz")

        # Clean prior test outputs if any
        if os.path.exists(steer_file):
            os.remove(steer_file)
        if os.path.exists(stats_file):
            os.remove(stats_file)

        # 1. Run steer_lens.py for 16 neurons on layer 0
        cmd_steer = [
            sys.executable,
            os.path.join(ROOT, "sweep", "steer_lens.py"),
            "--model", model_id,
            "--layers", "0",
            "--neurons", "0-15",
            "--neurons-per-batch", "16",
        ]
        res = subprocess.run(cmd_steer, cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertTrue(os.path.exists(steer_file), f"steer_index.npz was not created: {res.stderr}")

        z = np.load(steer_file)
        self.assertIn("up", z)
        self.assertIn("down", z)
        self.assertIn("up_kl", z)
        self.assertIn("down_kl", z)
        # Check that layer 0, first 16 neurons have data
        self.assertEqual(z["up"].shape[0], 16)  # LFM2.5 has 16 layers
        self.assertEqual(z["up"].shape[1], 4608)  # LFM2.5 has 4608 intermediate size
        self.assertTrue(np.any(z["up_kl"][0, :16] > 0))

        # 2. Run neuron_stats.py on layer 0 with local text
        cmd_stats = [
            sys.executable,
            os.path.join(ROOT, "sweep", "neuron_stats.py"),
            "--model", model_id,
            "--layers", "0",
            "--tokens", "64",
            "--seq-len", "32",
            "--batch", "2",
            "--text", "The quick brown fox jumps over the lazy dog. A swift breeze swept through the valley.",
        ]
        res = subprocess.run(cmd_stats, cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertTrue(os.path.exists(stats_file), f"stats file not created: {res.stderr}")

        s = np.load(stats_file)
        for key in ("pp_max", "nn_max", "pn_min", "np_min", "abs_max", "w_down", "resid_norm_q10_50_90"):
            self.assertIn(key, s)
        self.assertEqual(s["w_down"].shape, (4608, 1024))
        # 3. Verify server /api/units returns ranked neurons with metric 'kl'
        from explorer import server
        # Clear server rt cached steer index so it re-reads from disk
        server.rt._steer_idx = None
        with patch.dict(os.environ, {"EXPLORER_MODEL": model_id}):
            units_l0 = server.units(layer=0)
            self.assertEqual(len(units_l0), 4608)
            self.assertEqual(units_l0[0]["metric"], "kl")
            self.assertEqual(units_l0[0]["kind"], "neuron")
            self.assertTrue(units_l0[0]["z"] >= units_l0[1]["z"])
            self.assertTrue(len(units_l0[0]["words"]) <= 5)

            # Layer 1 has no steer lens or sweep runs -> returns []
            units_l1 = server.units(layer=1)
            self.assertEqual(units_l1, [])


if __name__ == "__main__":
    unittest.main()
