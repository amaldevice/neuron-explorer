# neuron explorer

Steering-first neuron explorer for transformer, linear-attention, Mamba, and convolution hybrid language models.
Every MLP neuron in a layer can be pushed up and down, activations inspected across arbitrary text, and causal
connections mapped over a live 3D view of the model.

![neuron explorer](explorer.png)

## Quickstart

```bash
uv sync
uv run python explorer/server.py        # http://localhost:8791
```

By default on Apple Silicon, the server loads `LiquidAI/LFM2.5-350M` on MPS (using ~700 MB RAM at >600 tok/s batched).
It automatically builds missing **Model artifacts** (`vocab.json`, `lens_index.npz`, `connections.npz`, and `act_scale.npz`)
under `explorer/cache/<slug>/` on startup.

### Environment Options

| Variable | Description | Default |
|---|---|---|
| `EXPLORER_MODEL` | **Subject model** to load (`LiquidAI/LFM2.5-350M`, `Qwen/Qwen3.5-0.8B`, `ibm-granite/granite-4.0-350m`, `ibm-granite/granite-4.0-h-350m`, `Qwen/Qwen3.5-4B`) | `LiquidAI/LFM2.5-350M` |
| `EXPLORER_AGENT` | CLI powering the assistant panel (`omp`, `pi`, or `claude`) | Auto-detected on `PATH` (`omp` → `pi` → `claude`) |
| `EXPLORER_AGENT_MODEL` | Model selector passed to the agent CLI (e.g. `openai/gpt-5.6-luna`, `claude-sonnet-5-5`) | CLI default / `EXPLORER_CLAUDE_MODEL` |
| `EXPLORER_NO_MODEL` | Run cache/metadata-only without loading PyTorch model weights | `0` |
| `PORT` | Local HTTP listen port | `8791` |

## Features

- **3D map**: Isometric visualization of decoder layers in perspective. Layer **Mixers** (`full_attention`, `linear_attention` GDN, `mamba` Mamba-2, and `conv` short-convolution) are rendered with truthful slab proportions. Neurons are laid out on a natural $\lceil\sqrt{N}\rceil \times \lceil\sqrt{N}\rceil$ sheet. Hover shows live logit and steer lenses; click opens the neuron; click a layer label to view face-on. Mode `writes` clusters neurons with similar down-projection columns.
- **Left panel**:
  - `net`: Top connections web of the focused neuron, followed by the layer's neuron list. When offline rollout sweeps are present, units are ranked by sweep word-cluster $z$-score; when sweeps are absent, units are automatically ranked by **Live Causal KL divergence** ($\max(\text{kl\_up}, \text{kl\_down})$ from `steer_index.npz`) with steer-lens token previews.
  - `run`: `fires` runs arbitrary user text through the model, capturing activations across all layers with a 1024-token safety clamp.
- **Neuron focus**:
  - `Chat`: 3-column multi-turn steered conversation (−0.8, baseline, +0.8) with `add` (including `residual_multiplier` compensation on Granite) and `mul` modes.
  - `Details`: Live logit and steer lenses, activation extremes, and direct-path upstream (`reads_from`) / downstream (`writes_into`) connection links with $z$-scores. When sweep data is present, displays word crumbs and dose tables.
  - `Examples`: Steered rollouts per prompt and dose (automatically hidden if no sweep rollouts exist).
- **Embeddings**: UMAP projection of a layer's rollouts (hidden when no sweep runs exist).
- **Agent panel**: In-app research assistant driven by **`omp`**, **`pi`**, or **`claude`** via the stdio MCP bridge (`explorer/agent_mcp.py`). Streams tokens live, displays tool calls, reads neuron cones, runs live steering/activations, controls UI selection/highlights, and supports `@file` attachments, turn interruption, and session rewind (`/api/agent/rewind`).

## Model Artifacts

All precomputed arrays bound to a Subject model are isolated under `explorer/cache/<slug>/` (`<slug> = model_id.lower().replace('/', '--')`):
- `vocab.json`: Model vocabulary strings
- `lens_index.npz`: Logit-lens top tokens per neuron
- `steer_index.npz`: Causal next-token push tokens and KL divergence from `sweep/steer_lens.py`
- `connections.npz`: Direct-path top-24 upstream and downstream links
- `act_scale.npz`: Typical activation ranges from FineWeb-Edu calibration
- `layout_writes.npz`: 2D UMAP layout snapped to the sheet grid
- `names.json`: Per-model user and agent neuron names

Every artifact loader validates array dimensions against `(n_layers, inter)` on load to guarantee shape safety.

## Causal Steer Lens & Offline Sweeps

1. **Fast Causal Steer Lens (Local MPS / Cloud GPU):**
   ```bash
   uv run python sweep/steer_lens.py --model LiquidAI/LFM2.5-350M
   ```
   Runs in ~4–6 minutes for all 16 layers of `LFM2.5-350M` on an Apple Silicon GPU (or ~1–2 minutes on free Kaggle/Colab T4 GPUs in `float16`/`bfloat16`), producing `steer_index.npz` and powering instant causal KL neuron rankings.

2. **Full Rollout Sweeps:**
   - **Local MPS**: Run adapter-backed subset sweeps for quick experimentation.
   - **Cloud GPU (Kaggle 2× T4 16 GB / Colab 1× T4 15 GB / Modal)**: Run multi-layer batched sweeps without compilation dependencies (see `docs/adr/0001-pytorch-mps-runtime-and-deferred-sweeps.md`). Optional high-throughput patches for the `RhizoNymph/vllm` steering fork are documented in `docs/adr/`.

## Architecture & Conventions

- Domain definitions: `GLOSSARY.md` (**Subject model**, **Agent**, **Model artifact**, **Mixer**)
- ADRs: `docs/adr/0001-pytorch-mps-runtime-and-deferred-sweeps.md`
- Source methodology: `NOTES.md` and `research_log.md`
