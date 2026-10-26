---
title: "Implications for neuron-explorer — what the current machinery can express"
tags: [research, neuron-explorer, steering, design-notes]
---

# Implications for neuron-explorer

This note connects the literature in the sibling notes to what the explorer
actually does today, and lists the research questions that fall out. It is a
design note, not a spec — the repo's own wayfinder issues (#1 and children) own
the spec.

## 1. The gap, stated precisely

The explorer's core primitive is **per-neuron steering at the MLP down-projection
input** (`explorer/runtime.py`, `_hook`): add, multiply, or ablate one neuron's
activation, measured in units of the layer's median residual norm.

The refusal literature's core primitive is **per-direction manipulation of the
residual stream**: project a direction out (ablation), add a direction in
(steering), or bake the projection into the weights (orthogonalization).

These are different objects. The explorer cannot currently express:

| Needed operation | Available today? |
| --- | --- |
| Add a *direction* $v$ to the residual stream at layer $\ell$ | No — only single-neuron additions via `W_down[:, n]` |
| Project a direction out of the residual stream | No |
| Project a direction out at a *subset* of layers | No |
| Project out a *set* of directions (cone / multi-D) | No |
| Bake the projection into the weights (orthogonalization) | No |
| Read the projection of a hidden state onto a direction | Partially — `read()` reads one neuron's activation; a projection onto an arbitrary vector is not exposed |
| Ablate an SAE feature (feature-level, not direction-level) | No — the explorer has no SAE |
| Steer *during* training (preventative steering) | No |

The nearest thing: `Runtime.down_col(layer, n)` returns
`W_down[:, n]`, a residual-space direction that a single neuron writes. So
`∑_n α_n W_down[:, n]` is the explorer's current way to add an arbitrary
residual direction — but only in the span of MLP down columns, and only
one neuron at a time in the live path (`steer_multi` does support several at
once, but they must be neurons).

## 2. What the literature says about the right shape of the next step

Three findings point at concrete additions, in decreasing order of confidence:

### 2.1 A direction-level steer/ablate, as a sibling of the neuron push

The explorer's existing `mode` vocabulary is `add` / `mul`. A
`direction` mode needs a vector $v \in \mathbb{R}^{d_{model}}$, a layer, a
coefficient, and a sign. Mechanically it is a hook on the residual stream
(output of a block, or input of the next), not on `down_proj`.

Practical note: the explorer already has a `_resid_hook` on each block
(`layer.register_forward_hook(self._resid_hook(i))`) that only records the
residual norm. That is a natural place to extend — the hook already sees the
post-block residual.

The dose unit question is real and already answered elsewhere in the repo:
`research_log.md` records the convention that `add` doses are in units of the
median residual norm leaving the layer (`--dose-unit rel`). The same unit for a
direction dose keeps the two comparable, which matters if a person is comparing
"push this neuron by +2" against "push the refusal direction by +2".

### 2.2 The honest caveat: refusal is not one direction, and not a neuron set either

The literature is split on dimensionality, and the explorer should not hard-code
either answer:

- Arditi: 1-D suffices, across 13 models.
- Wollschläger: a *cone* of directions, all controlling refusal; orthogonality
  does not imply independence.
- Piras (SOM): multiple directions beat one.
- Joad: 11 refusal categories → geometrically distinct directions, but a shared
  one-dimensional *control knob*.
- RFM-AGOP: multi-dimensional subspaces, ≥3 directions needed for Qwen3 8B at
  50% ASR per the community wiki.
- SAE work: refusal is a *set* of features, and harmful-content features are
  distinct from refusal features.

The defensible design is a *list* of directions with a per-direction weight,
which degrades gracefully: one direction is the Arditi case, several is the
cone case, and the interface does not assert a dimensionality claim.

### 2.3 Persona as a second axis, not a footnote

*Refusal Lives Downstream of Persona* reports compliant-persona steering dropping
Llama's refusal rate from 97% to 2% — a bigger lever than the refusal direction
itself, at least in that setup. And the persona-vectors pipeline produces a
*compliant* persona direction the same way it produces an *evil* one: a
contrastive pair of system prompts, difference-in-means, one vector per layer.

That maps unusually well onto the explorer: the persona direction can be
extracted with a difference-in-means over two system prompts, and the explorer
already has the machinery to read and steer a single direction once it can hold
one. This is a cheap, self-contained experiment with a clear success criterion.

## 3. Candidate experiments, cheapest first

Each states the success criterion, since the repo's AGENTS.md requires it.

**E1 — Does the explorer's subject model have a refusal direction at all?**
Extract a difference-in-means vector between harmful (AdvBench) and harmless
(Alpaca) prompts, per layer, at the post-instruction token. Success: a direction
whose projection separates the two prompt sets at some layer, and ablating it
reduces refusal rate on a held-out harmful set. Cost: light (forward passes
only, no training). This is the *minimum* to claim the explorer can see refusal.

**E2 — Ablate vs. add, at the layer band the literature points to.**
Arditi's LessWrong reply says layers 14–17 of Qwen 1.8B do the work; Wollschläger
and the community say mid-to-late; Zhong & Li say a *late* window gates
expression. Success: a layer sweep showing refusal drops for a band and not for
the whole network — i.e. reproducing the locality result.

**E3 — Is the direction a cone?**
Extract several directions (SOM-style or top-$k$ PCA on the difference matrix)
and check whether ablating the top-1 vs top-$k$ differs at the explorer's model
size. Success: measurable ASR/refusal difference between top-1 and top-$k$
ablation, or a clear negative result (top-1 is enough at this scale). Either
outcome is informative and cheap.

**E4 — Persona gating.**
Extract a "compliant persona" direction from two system prompts; steer it and
measure refusal rate. Success: a refusal-rate drop that survives a refusal-
direction ablation (i.e. persona steering works even with the refusal direction
ablated), reproducing the "gated, not computed" claim.

**E5 — Covert non-compliance.**
The community's most repeated failure mode: after ablation the model stops
emitting refusal templates but still hedges. Success: an evaluation that
separates *refusal-string rate* from *LLM-judged refusal rate* on the same
outputs, and shows the gap. This is the experiment that would make the
explorer's refusal claims credible rather than template-matching.

**E6 — Monitoring.**
Project the final prompt token onto the persona vector and correlate with
downstream behavior. The persona-vectors paper reports $r = 0.75$–$0.83$ for
system-prompt variation, with weaker correlation once prompt type is
controlled. Success: a correlation, and an honest report of how much of it
survives controlling for prompt type.

## 4. Evaluation protocol to reuse (from the sibling notes)

If the explorer is to make refusal claims, it needs all of:

- refusal rate on a harmful set (JailbreakBench 100, or AdvBench),
- over-refusal on a benign set (XSTest, OR-Bench) — a refusal *removal* tool
  that ignores over-refusal is not a useful tool,
- a capability check (MMLU, GSM8K) — the literature's consistent finding is
  that abliteration costs some TruthfulQA and some capability, and that the
  drop shows up in coherence before it shows up in MMLU,
- KL from the base model on harmless prompts (the community's own collateral
  measure),
- a coherence check separate from all of the above.

**Evaluation-set choice for a sub-1B subject model.** The explorer's subject
models are Qwen3.5-0.8B, LFM2.5-350M, Granite-4.0-350M. Two caveats to check
before trusting any refusal number on them: base-model refusal rates on these
sizes are often much lower than on 7B chat models, and small models may have
little refusal direction to find at all. The correct first check is a
*baseline refusal rate* measurement; if the model refuses almost nothing, there
is nothing to ablate and E1–E4 are moot for that subject model. This is a
one-hour check that should precede everything else.

## 5. Sizing note

The explorer's own subject-model choices are unusually well suited to this
research: refusal-direction work needs only forward passes and hooks, and the
repo's Apple-Silicon findings (issue #2) established that steered rollouts and
activation capture run on MPS in bf16 for exactly these models. No GPU rental,
no training, no dataset downloads beyond a few small prompt files. The heavy
part of the abliteration literature (weight orthogonalization, Optuna search
over 200 trials, Heretic's benchmark matrix) is not needed to answer the
mechanistic questions.

## 6. What I did not do

- Ran nothing. No model was loaded, no direction was extracted, no number in
  these notes is locally measured.
- Did not open the repo's wayfinder issues to propose a new ticket; the map
  (#1) and its children are the place for that, and the open children are
  about adapters, sweep paths and agent backends, none of which this research
  changes.
- Did not check whether `docs/research/` conflicts with anything existing;
  `docs/` currently holds only `docs/agents/`.
