# 0001. Keep Runtime on PyTorch MPS and Bridge Missing Sweeps with Steer Lens

Status: accepted

## Context and Decision

On an 8 GB Apple Silicon Mac (`Mac17,5`, A18 Pro 5-core GPU via Metal/MPS, no NVIDIA CUDA GPU), we need `neuron-explorer` to run live exploration on sub-1B Subject models (`Qwen/Qwen3.5-0.8B`, `LiquidAI/LFM2.5-350M`, `ibm-granite/granite-4.0-350m`, `ibm-granite/granite-4.0-h-350m`) and have a practical path for causal neuron ranking and later rollout sweeps. Benchmarks on this machine (`research/sweep-on-apple-silicon` @ `ecdff7e` and `research/fork-arch-support` @ `65a5ea2`) showed:

- **PyTorch `transformers` on MPS (`bfloat16`)** runs steered generation + `mlp_in` capture + top-5 logprobs across all four models with zero CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK=0`) at **624.4 tok/s** (`LFM2.5-350M`, $B=64$), **416.2 tok/s** (`Granite-4.0-350M`, $B=64$), **43.4 tok/s** (`Qwen3.5-0.8B`, $B=16$), and **30.6 tok/s** (`Granite-4.0-H-350M`, $B=16$).
- **`vllm-project/vllm-metal` and CPU `RhizoNymph/vllm` do not work** for steering: `MetalModelRunner` bypasses `GPUModelRunner` and `torch.ops.vllm.apply_steering`/`capture_residual`, `VLLM_METAL_BACKEND=mps` rejects multimodal/MoE/GDN/Mamba-2 layers and lacks `"MPS"` custom-op dispatch keys, and `RhizoNymph/vllm` has no steering hooks wired in `lfm2.py` or `granitemoehybrid.py`.
- **`mlx-lm` with `nn.Module` wrapping** works and is faster on hybrid recurrent models (**124.5 tok/s** on `Qwen3.5-0.8B`, **196.0 tok/s** on `Granite-4.0-H-350M` via native Metal kernels), but would require maintaining a second model-loading and hook stack alongside PyTorch.

We therefore decide:

1. **Single PyTorch runtime (`AutoModelForCausalLM` + `ModelAdapter`)** for the live explorer on MPS/CUDA/CPU, with no `mlx-lm` or `vllm-metal` dependency in v1.
2. **`sweep/steer_lens.py` on MPS as the fast local causal signal**: a single-step prefill push ($\pm 0.8$ residual-norm units across 4 short contexts) computes per-neuron `steer_up`, `steer_down`, `kl_up`, and `kl_down` into `explorer/cache/<slug>/steer_index.npz` in ~4–6 minutes for an entire 350M model on the local A18 Pro GPU (or ~1–2 minutes on a free Colab/Kaggle T4).
3. **Two-tier rollout sweeps post-v1 using the same PyTorch `ModelAdapter`**:
   - **Local MPS** for quick single-layer or 448-unit subset sweeps on `LFM2.5-350M` (~14 min/layer) and `Granite-4.0-350M` (~21 min/layer).
   - **Free Cloud GPUs (Kaggle 2× T4 16 GB / Colab 1× T4 15 GB)** for multi-layer or full sweeps, running the same adapter-backed PyTorch batched sweep script on `cuda` (supporting `float16` on Turing T4 where native `bfloat16` Tensor Cores are absent, as well as `bfloat16` on Ampere+), with optional `RhizoNymph/vllm` CUDA support via the 35-line pure-Python patches documented on branch `research/fork-arch-support`.
