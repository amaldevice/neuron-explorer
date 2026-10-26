# Neuron Explorer

Mechanistic interpretability workbench for steering, reading, and mapping MLP neurons of a language model and investigating them with a tool-using assistant.

## Language

**Subject model**:
The language model whose MLP neurons are inspected, steered, and measured in the explorer (for example `Qwen/Qwen3.5-0.8B`, `LiquidAI/LFM2.5-350M`, or `ibm-granite/granite-4.0-350m`).
_Avoid_: Model (unqualified), target model

**Agent**:
The external assistant (driven through a CLI such as `claude`, `omp`, or `pi`) that powers the explorer's assistant panel and calls its MCP tools.
_Avoid_: Claude (when referring to the role rather than the Anthropic CLI backend)

**Model artifact**:
A precomputed file bound to the layer and neuron dimensions of one specific Subject model, including `lens_index.npz`, `steer_index.npz`, `connections.npz`, `act_scale.npz`, `layout_writes.npz`, `vocab.json`, per-layer stats, and sweep runs.
_Avoid_: Cache (when referring to Subject-model-bound data rather than ephemeral runtime state)


**Mixer**:
The sequence-mixing sub-layer preceding the MLP in each decoder block (`full_attention`, `linear_attention`, `mamba`, or `conv`).
_Avoid_: Attention layer (when referring to non-attention sequence mixers such as Mamba or short convolution)