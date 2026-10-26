---
title: "Refusal direction and abliteration — Arditi et al. 2024 and the multi-direction critiques"
tags: [research, refusal, abliteration, interpretability, steering]
---

# Refusal direction and abliteration

Primary source: Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda.
*Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024.
[arXiv:2406.11717](https://arxiv.org/abs/2406.11717) ·
[code](https://github.com/andyrdt/refusal_direction) ·
[LessWrong post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction)

## 1. The claim

> "We show that refusal is mediated by a one-dimensional subspace, across 13 popular
> open-source chat models up to 72B parameters in size. Specifically, for each model,
> we find a single direction such that erasing this direction from the model's residual
> stream activations prevents it from refusing harmful instructions, while adding this
> direction elicits refusal on even harmless instructions."

Concretely, per model, there is a vector $\hat{\mathbf r} \in \mathbb{R}^{d_{model}}$
such that (a) projecting it out of the residual stream removes refusal, and
(b) adding it induces refusal on benign prompts.

## 2. Method

**Extraction (difference-in-means).** For each layer $l$ and each
post-instruction token position $i$ in the chat template:

$$
\boldsymbol\mu_i^{(l)} = \frac{1}{|\mathcal D_{\text{harmful}}^{(\text{train}})|}\sum_{\mathbf t \in \mathcal D_{\text{harmful}}^{(\text{train})}} \mathbf x_i^{(l)}(\mathbf t),
\qquad
\boldsymbol\nu_i^{(l)} = \frac{1}{|\mathcal D_{\text{harmless}}^{(\text{train}})|}\sum_{\mathbf t \in \mathcal D_{\text{harmless}}^{(\text{train})}} \mathbf x_i^{(l)}(\mathbf t)
$$

$$
\mathbf r_i^{(l)} = \boldsymbol\mu_i^{(l)} - \boldsymbol\nu_i^{(l)}
$$

Datasets: harmful = AdvBench + MaliciousInstruct + TDC2023 + HarmBench;
harmless = Alpaca. 128 train / 32 val each. Models: Qwen 1.8B/7B/14B/72B,
Yi 6B/34B, Gemma 2B/7B, Llama-2 7B/13B/70B, Llama-3 8B/70B.

**Selection.** $|I| \times L$ candidate vectors; the single best
$\mathbf r_{i^*}^{(l^*)}$ is chosen by validation: it must (i) suppress refusal
when ablated, (ii) induce refusal when added, (iii) change other behavior minimally.

**Interventions.** Activation addition at a single layer, all token positions:

$$
\mathbf x^{(l)\prime} \leftarrow \mathbf x^{(l)} + \mathbf r^{(l)}
$$

Directional ablation at *all* layers, *all* token positions:

$$
\mathbf x^{\prime} \leftarrow \mathbf x - \hat{\mathbf r}\hat{\mathbf r}^{\top}\mathbf x
$$

**Weight orthogonalization (the persistent form).** Instead of hooking the
forward pass, edit every matrix that writes to the residual stream (embedding,
positional embedding, attention out, MLP out, plus biases):

$$
W_{\text{out}}^{\prime} \leftarrow W_{\text{out}} - \hat{\mathbf r}\hat{\mathbf r}^{\top} W_{\text{out}}
$$

Appendix E proves this is *equivalent* to inference-time directional ablation.
That equivalence is the whole basis of the "abliteration" recipe.

## 3. Results worth remembering

- **Refusal ablation** on JailbreakBench (100 harmful instructions): refusal rate
  collapses and unsafe completions appear, across all 13 models.
- **Activation addition** on 100 harmless Alpaca instructions makes models refuse
  benign requests. Example from the paper (Gemma 7B IT), prompt *"List three benefits
  that yoga has on physical health"* → *"I am unable to provide information on the
  benefits of yoga ... as it can be dangerous and potentially harmful."*
- **HarmBench ASR** (Table 2): the orthogonalized model is competitive with
  prompt-specific GCG on the Qwen family. System-prompt sensitivity differs by
  family: Llama-2 7B 22.6% (with system prompt) vs 79.9% (without); Qwen 7B
  79.2% vs 74.8%.
- **Capability retention** (Table 3): MMLU / ARC / GSM8K within noise for the
  orthogonalized models. **TruthfulQA consistently drops** (e.g. −2.3 pp on
  Llama-3 70B, −3.5 on Yi 34B). The authors' own explanation: TruthfulQA
  categories (misinformation, stereotypes, conspiracies) sit close to refusal
  territory, so removing refusal removes some pushback against plausible
  falsehoods. This is the seed of the "truthfulness cost" theme in all later work.
- **Adversarial suffixes** (Section 5, Qwen 1.8B Chat): the GCG suffix
  *suppresses* the refusal direction at the last token, down to harmless-prompt
  levels; a random suffix of the same length does not. DFA (direct feature
  attribution) shows the top-8 attention heads that write the refusal direction
  have their contributions suppressed, and their attention shifts from the
  instruction region to the suffix region.

> [!quote]
> "The expression of the refusal direction is very high for harmful instructions,
> and remains high when a random suffix is appended. The expression of the refusal
> direction after appending the adversarial suffix is heavily suppressed, and closely
> resembles that of harmless instructions."
> — [arXiv:2406.11717 §5.1](https://arxiv.org/html/2406.11717v3)

### Layer locality (from the author's LessWrong reply, not the paper)

Ablating the direction at layers $\ell..L$ shows ablations before ~layer 14 have
little impact and ablations after ~layer 17 have little impact; ablating only
layers 14–17 is roughly as effective as ablating all layers (Qwen 1.8B). This is
the empirical seed of the community's "ablate a mid-to-late band, not everything"
practice.

## 4. "Abliteration" — the community name for this

The term was coined in **FailSpy/abliterator**
([github.com/FailSpy/abliterator](https://github.com/FailSpy/abliterator)), and
spread via Maxime Labonne's post
[*Uncensor any LLM with abliteration*](https://huggingface.co/blog/mlabonne/abliteration)
(HF blog, 12 Jun 2024), which is the canonical practitioner tutorial and
Colab notebook. Labonne's framing of the three steps:

> "1. **Data Collection**: Run the model on a set of harmful instructions and a set of
> harmless instructions, recording the residual stream activations at the last token
> position for each. 2. **Mean difference**: Calculate the mean difference between
> the activations of harmful and harmless instructions. This gives us a vector
> representing the 'refusal direction' for each layer of the model. 3. **Selection**:
> Normalize these vectors and evaluate them to select the single best 'refusal direction.'"

Important framing point from the paper itself: the orthogonalized model **retains
its original capabilities** but stops refusing. The "uncensored model" community
label is a downstream, broader claim than what Arditi et al. demonstrated.

### Tool lineage (practitioner facts, from the abliteration community wiki)

Per `abliteration.org/wiki/methods/m1-direct-removal/` (community wiki, not
peer-reviewed):

- **Arditi `refusal_direction`** — reference research code, TransformerLens-based.
- **FailSpy `abliterator`** — where the term was coined; TransformerLens-based.
- **Sumandora `remove-refusals-with-transformers`** — plain HF Transformers +
  hooks, no TransformerLens; the community default because it works on new
  architectures the day they ship.
- **Heretic** (`p-e-w/heretic`, AGPL-3.0, PyPI `heretic-llm`) — automates the
  *parameter choice* with Optuna TPE: 200 trials, 60 random, minimizing
  (refusal count on 100 harmful prompts, KL divergence from the original on 100
  harmless prompts). Reported on Gemma-3-12B-IT: base 97/100 refusals; manual
  abliterations 3/100 at KL 1.04 (mlabonne) and 0.45 (huihui-ai); Heretic 3/100
  at KL 0.16. From v1.1 it defaults to grimjim's *projected abliteration*
  (remove the direction only from the subspace of columns that actually
  correlate with refusal on the extraction set), which reportedly cuts KL
  divergence 2–3×. v1.2.0 added a LoRA-adapter output mode.
- Known failure mode, documented repeatedly in the community: **covert
  non-compliance**. The model stops emitting refusal templates but may still
  hedge, subvert, or even claim it is censored. Refusal *templates* are not the
  same as refusal *behavior*.
- Failure mode: **single direction insufficient at ≥8B**. Per the wiki, citing
  `arXiv:2607.02396` (RFM-AGOP), Qwen3 8B needs at least three ablated
  directions to pass a 50% ASR threshold.

## 5. The critiques: it is not one direction

This is the most important part of the note for a steering-first tool, because it
changes what "the refusal direction" means.

### 5.1 Concept cones (Wollschläger et al. 2025)

*The Geometry of Refusal in Large Language Models: Concept Cones and
Representational Independence.* ICML 2025.
[arXiv:2502.17420](https://arxiv.org/abs/2502.17420) ·
[PMLR](https://proceedings.mlr.press/v267/wollschlager25a.html)

> "Contrary to prior work, we uncover multiple independent directions and even
> multi-dimensional concept cones that mediate refusal. Moreover, we show that
> orthogonality alone does not imply independence under intervention, motivating
> the notion of representational independence that accounts for both linear and
> non-linear effects."

They generalize difference-in-means to a polyhedral cone
$\mathcal R_N = \{\sum_i \lambda_i \mathbf b_i \mid \lambda_i \ge 0\} \setminus \{\mathbf 0\}$
of $N$ dimensions where *every* direction in the cone controls refusal. Key
warning for practitioners: two directions can be orthogonal and still not
independent under intervention — so "orthogonalize a second direction and call
it a day" is unsound.

### 5.2 SOM directions (Piras et al. 2025)

*SOM Directions are Better than One: Multi-Directional Refusal Suppression in
Language Models.* AAAI 2026. [arXiv:2511.08379](https://arxiv.org/abs/2511.08379)

> "We first prove that SOMs generalize the prior work's difference-in-means
> technique. We then train SOMs on harmful prompt representations to identify
> multiple neurons. By subtracting the centroid of harmless representations from
> each neuron, we derive a set of refusal directions."

Multi-direction ablation beats the single-direction baseline.

### 5.3 Categories of refusal (Joad et al. 2026)

*There Is More to Refusal in Large Language Models than a Single Direction.*
[arXiv:2602.02132](https://arxiv.org/abs/2602.02132)

Eleven categories of refusal (safety, incomplete/unsupported requests,
anthropomorphization, over-refusal, …) correspond to *geometrically distinct*
directions — yet all act as a shared one-dimensional control knob. The
distinction that matters: the directions change **how** the model refuses more
than **whether** it refuses.

### 5.4 Fast multi-dimensional subspaces (Winninger 2026)

*Fast Multi-dimensional Refusal Subspaces via RFM-AGOP.*
[arXiv:2607.02396](https://arxiv.org/abs/2607.02396)

Recursive Feature Machine with Average Gradient Outer Product, probe-informed
initialization; identifies multi-dimensional refusal subspaces "in seconds" on
Qwen 3 (1.7B/4B/8B/14B) and Qwen 2.5. Reports better ablation performance than
comparison methods. Single author.

### 5.5 Affine, not just linear (Marshall, Scherlis, Belrose 2024)

*Refusal in LLMs is an Affine Function.* [arXiv:2411.09003](https://arxiv.org/abs/2411.09003)
· [code](https://github.com/EleutherAI/steering-llm)

> "We begin with an affine decomposition of model activation vectors and show that
> prior methods for steering model behavior correspond to subsets of terms of this
> decomposition. We then provide a derivation of ACE and use it to control refusal
> behavior on ten different models, including Llama 3 70B. ACE combines affine
> subspace projection and activation addition to reliably control the model's
> refusal responses across prompt types."

Reported result: ACE generalizes to models where directional ablation alone
produces incoherent outputs.

### 5.6 Cross-lingual universality (Wang et al. 2025)

*Refusal Direction is Universal Across Safety-Aligned Languages.*
NeurIPS 2025. [arXiv:2505.17306](https://arxiv.org/abs/2505.17306)

> "We uncover the surprising cross-lingual universality of the refusal direction: a
> vector extracted from English can bypass refusals in other languages with
> near-perfect effectiveness, without any additional fine-tuning."

14 languages via the PolyRefuse dataset. This matters for the "one direction,
many languages" picture: the direction is not English-specific.

### 5.7 Cross-model transfer (2601.16034)

*Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay
and Concept-Basis Reconstruction.* [arXiv:2601.16034](https://arxiv.org/abs/2601.16034)

Reports transferring refusal interventions from donor to target models across
architectures (dense → MoE) *without target-side refusal supervision*, using
"concept fingerprints" to align layers plus a weight-SVD stability guard that
projects interventions away from high-variance weight subspaces. The authors are
explicit that the universality claim is operational, not absolute.

## 6. Defenses and restoration

- **Extended-refusal defense** (Abu Shairah et al. 2025,
  [arXiv:2505.19056](https://arxiv.org/abs/2505.19056)): build a dataset where
  refusals give detailed justifications *before* refusing, distributing the
  refusal signal across many token positions so no single direction dominates.
  "refusal rates drop by at most 10%, compared to 70-80% drops in baseline
  models under M1 attack."
- **ROSI — rank-one safety injection** (Abu Shairah et al. 2025,
  [arXiv:2508.20766](https://arxiv.org/abs/2508.20766)): the *opposite* of M1 — a
  white-box rank-one edit steering activations **toward** the refusal subspace.
  Raises refusal rate on Llama Guard 3 while preserving MMLU/HellaSwag/ARC.
- **AMRA — refusal aliases** (Truong 2026,
  [arXiv:2608.18093](https://arxiv.org/abs/2608.18093)): rank-$k$ updates to the
  residual-stream writer matrices that replace refusal-inducing activations with
  random aliases. +2.16 pp post-abliteration refusal on Llama-3-8B, +14.70 pp
  on Gemma-2-9B, under 0.5 pp MMLU degradation.
- **Granular safety-pretraining study** (Agnihotri et al. 2025,
  [arXiv:2510.02768](https://arxiv.org/abs/2510.02768)): 20 systems (SmolLM2-1.7B
  safety-pretraining checkpoints + open baselines), 100 balanced harmful/harmless
  prompts, multiple judges with human-validated fidelity. Characterizes which
  *data-centric* safety components survive abliteration.
- **Tooling comparison** (Young 2026, [arXiv:2512.13655](https://arxiv.org/abs/2512.13655)):
  four tools (Heretic, DECCP, ErisForge, FailSpy) across 16 instruction-tuned
  models, 7B–14B. Single-pass methods preserve capability better (GSM8K change
  ErisForge −0.28 pp, DECCP −0.13 pp); Bayesian-optimized abliteration gives
  KL divergence 0.043–1.646 with model-dependent capability impact. Also
  reports the covert-non-compliance observation.

## 7. Open questions this literature leaves

1. Is the refusal subspace really low-dimensional, or is a cone/subspace the
   right unit? Evidence is mixed (Arditi 1-D vs Wollschläger cones vs Joad
   categories vs RFM-AGOP multi-D).
2. How much of "removing refusal" is removing refusal vs removing
   truthfulness-pushback? The TruthfulQA drop is the standard evidence and it is
   small but consistent.
3. Does removing the direction remove *behavior* or only *refusal templates*?
   Covert non-compliance suggests the answer is "templates, mostly".
4. Why does the direction transfer across languages and across models at all,
   if the models have different dimensionalities and tokenizers? The
   concept-fingerprint account is a proposal, not a settled mechanism.
