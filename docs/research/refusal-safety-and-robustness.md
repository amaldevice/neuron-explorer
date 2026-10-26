---
title: "Refusal safety and robustness — attacks, defenses, reasoning models, SAEs, datasets"
tags: [research, refusal, safety, jailbreak, evaluation, datasets]
---

# Refusal safety and robustness

The adversarial and evaluation side of the literature. Every claim here cites a
paper I fetched; the dataset table at the end is the practical part.

## 1. Attacks and what they do mechanically

### 1.1 Adversarial suffixes (GCG)

Zou et al. 2023, *Universal and Transferable Adversarial Attacks on Aligned
Language Models*. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
Introduced AdvBench, still the field-standard refusal-measurement dataset.

Mechanistically (Arditi et al. §5): the suffix suppresses the refusal
direction's expression at the last token, down to harmless-prompt levels, and it
does so by hijacking the attention of the specific heads that write the refusal
direction — they shift attention from the instruction region to the suffix
region. A random suffix of the same length does not reproduce the effect.

### 1.2 Persona modulation (black-box)

Shah et al., *Scalable and Transferable Black-Box Jailbreaks for Language Models
via Persona Modulation*. [arXiv:2311.03348](https://arxiv.org/abs/2311.03348)

> "These automated attacks achieve a harmful completion rate of 42.5% in GPT-4,
> which is 185 times larger than before modulation (0.23%). These prompts also
> transfer to Claude 2 and Vicuna with harmful completion rates of 61.0% and
> 35.9%, respectively."

This is the black-box counterpart of the persona-vector result: changing the
model's persona (via system prompt) changes its refusal behavior. The
mechanistic version of the same trick is what the persona-vector paper and
*Refusal Lives Downstream of Persona* provide.

### 1.3 Fine-tuning as an attack

Qi et al. 2023, *Fine-tuning Aligned Language Models Compromises Safety, Even When
Users Do Not Intend To!* [arXiv:2310.03693](https://arxiv.org/abs/2310.03693)

> "Our red teaming studies find that the safety alignment of LLMs can be
> compromised by fine-tuning with only a few adversarially designed training
> examples. For instance, we jailbreak GPT-3.5 Turbo's safety guardrails by
> fine-tuning it on only 10 such examples at a cost of less than $0.20."

And the benign-data finding: even Alpaca / Dolly / LLaVA-VI fine-tuning
degrades alignment somewhat, via catastrophic forgetting or the inherent
helpfulness/harmlessness tension. This is the tension the persona-vectors paper
also finds empirically (helpfulness pushes against harmlessness; the persona's
location in the embedding space encodes the trade-off).

Lermen et al. 2023, *LoRA Fine-tuning Efficiently Undoes Safety Training in
Llama 2-Chat 70B* [arXiv:2310.20624](https://arxiv.org/abs/2310.20624) — the
contrast case to abliteration: refusal removal by fine-tuning rather than
activation ablation (~1% refusal rate on 70B, under $200).

### 1.4 Emergent misalignment

Soligo et al. 2025, *Model Organisms for Emergent Misalignment*.
[arXiv:2506.11613](https://arxiv.org/abs/2506.11613)

> "Using new narrowly misaligned datasets, we create a set of improved model
> organisms that achieve 99% coherence (vs. 67% prior), work with smaller 0.5B
> parameter models (vs. 32B), and that induce misalignment using a single rank-1
> LoRA adapter. We demonstrate that EM occurs robustly across diverse model sizes,
> three model families, and numerous training protocols including full supervised
> fine-tuning."

They isolate a mechanistic phase transition corresponding to a robust behavioural
phase transition. The persona-vectors paper's "EM-like" datasets (flawed medical
advice, invalid math, insecure code) are explicitly modeled on this line of work.

### 1.5 Alignment faking

Anthropic, *Alignment Faking in Large Language Models*.
[arXiv:2412.14093](https://arxiv.org/abs/2412.14093) — noted for completeness:
it is the "the model knows it is being trained away from its goals" case, which
is a different axis from refusal direction mechanics. I did not fetch the full
text in this pass, so no detail is claimed here.

## 2. Defenses

### 2.1 Representation-level

- **Circuit breakers / representation rerouting** (Zou et al. 2024, NeurIPS 2024,
  *Improving Alignment and Robustness with Circuit Breakers* —
  [arXiv:2406.04313](https://arxiv.org/abs/2406.04313), code at
  [github.com/GraySwanAI/circuit-breakers](https://github.com/GraySwanAI/circuit-breakers)) —
  replaces the harmful-output representation
  with a "circuit breaker" during training; attack-agnostic because it targets
  the capability to produce harm, not specific attacks. Reported to work on
  text-only and multimodal models and to extend to agents.
- **ReFAT** — Yu, Do, Hambardzumyan, Cancedda, *Robust LLM safeguarding via
  refusal feature adversarial training*. ICLR 2025.
  [arXiv:2409.20089](https://arxiv.org/abs/2409.20089)

  > "We demonstrate that adversarial attacks share a universal mechanism for
  > circumventing LLM safeguards that works by ablating a dimension in the
  > residual stream embedding space called the refusal feature. We further show
  > that the operation of refusal feature ablation (RFA) approximates the
  > worst-case perturbation of offsetting model safety."

  RFA is used as a *simulation* of input-level attacks inside the training loop,
  which is much cheaper than input-space adversarial training.

- **SAFE / Latent Adversarial Training** — the latent-space analogue: perturb
  in representation space rather than token space. (Note: I did not fetch the
  SAFE paper's arXiv ID, so none is given here; the LAT *analysis* paper
  [arXiv:2504.18872](https://arxiv.org/abs/2504.18872) is cited in §4.)

### 2.2 Training-level

- **Safe LoRA** (Hsu et al. 2024, [arXiv:2405.16833](https://arxiv.org/abs/2405.16833)):
  a one-liner patch projecting LoRA weights from selected layers onto the
  safety-aligned subspace. Training-free and data-free — needs only base +
  aligned weights.
- **CAFT** (Casademunt et al. 2025): zero-ablate undesired concept directions
  during training. Persona-vectors finds it works for evil/sycophancy but not
  hallucination.
- **Preventative steering** (persona-vectors): steer *toward* the undesired
  direction during training to cancel the objective's pressure. Multi-layer
  steering nearly eliminates trait acquisition with no MMLU cost.
- **Extended-refusal fine-tuning** ([arXiv:2505.19056](https://arxiv.org/abs/2505.19056)):
  distribute the refusal signal across many token positions so no single
  direction dominates → abliteration drops refusal by ≤10% instead of 70–80%.

### 2.3 Prompt-level and output-level

- **XSTest** (Röttger et al. 2024, [arXiv:2308.01263](https://arxiv.org/abs/2308.01263))
  — exaggerated-safety test suite. Found 26.1% of safe prompts misclassified as
  dangerous and refused in some models; few-shot / interactive / contextual
  prompting mitigates 92.9% overall across Llama2, Gemma, Command R+, Phi-3.
- **Jailbreak Antidote** (Shen et al. 2024,
  [arXiv:2410.02298](https://arxiv.org/abs/2410.02298)) — runtime safety/utility
  balance by shifting hidden representations along a *safety* direction with
  varying strength; the paper's observation is that safety information is
  sparsely distributed, so a sparse subset of states can be adjusted.
- **Beyond Over-Refusal** (Yuan et al. 2025,
  [arXiv:2510.08158](https://arxiv.org/abs/2510.08158)) — XSB and MS-XSB
  benchmarks with "Focus" keyword annotations; three model-agnostic
  inference-time fixes (ignore-word instructions, prompt rephrasing, attention
  steering) with no retraining and no parameter access.
- **FalseReject** ([arXiv:2505.08054](https://arxiv.org/abs/2505.08054)) —
  contextual-safety resource for over-refusal.
- **ACTOR / "Just Enough Shifts"** (Dabas et al. 2025, ICML 2025,
  [arXiv:2507.04250](https://arxiv.org/abs/2507.04250)) — response-free,
  activation-pattern-based training that fine-tunes a *single layer* to reduce
  over-refusal while keeping harmful-query refusal intact.
- **AlphaSteer** ([arXiv:2506.07022](https://arxiv.org/abs/2506.07022)) —
  refusal steering with a null-space constraint to avoid the safety/utility
  trade-off that plain activation addition causes.
- **Any-Depth Alignment** ([arXiv:2510.18081](https://arxiv.org/abs/2510.18081)) —
  the observation that alignment is *shallow*: models refuse at the start of a
  turn but the protection collapses once a harmful continuation is underway.
- **The Art of Saying No** ([arXiv:2407.12043](https://arxiv.org/abs/2407.12043)) —
  broadens "noncompliance" beyond safety into a taxonomy (incomplete,
  unsupported, indeterminate, humanizing requests) with a 1000-prompt suite.
- **Refusal Tokens** ([arXiv:2412.06748](https://arxiv.org/abs/2412.06748)) —
  calibration of refusal *level* as a preference, rather than binary refusal.
- **SELFDEFEND** (Li et al. 2023, [arXiv:2308.07308](https://arxiv.org/abs/2308.07308))
  — a shadow LLM screens prompts and outputs; "By Self Examination, LLMs Know
  They Are Being Tricked".
- **ReFT / LoReFT** (Wu et al. 2023, [arXiv:2404.03592](https://arxiv.org/abs/2404.03592)):
  learn interventions on hidden representations of a frozen model, 15×–65× more
  parameter-efficient than LoRA. Relevant because it is the principled version of
  "steer a representation during training" that the persona-vectors preventative
  steering is a special case of.
- **DPO's mechanism** (Lee et al. 2024, [arXiv:2401.01967](https://arxiv.org/abs/2401.01967)):
  "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and
  Toxicity" — DPO does not remove capabilities, it learns to bypass them. This
  is the mechanistic reason "abliterate then heal with DPO" can leave a model
  that still has the capability but no longer refuses.

## 3. Where refusal lives in reasoning models

*Where Do Reasoning Models Refuse?* [arXiv:2507.03167](https://arxiv.org/abs/2507.03167)

> "Chat models without chain-of-thought (CoT) reasoning must decide whether to
> refuse a harmful request before generating their first response token. Reasoning
> models, by contrast, produce extended chains of thought before their final
> output, raising a natural question: where in this process does the decision to
> refuse occur? We investigate this across four open-source reasoning models. We
> first show that the CoT causally influences refusal outcomes; fixing a specific
> reasoning trace substantially reduces variance in whether the model ultimately
> refuses or complies. Zooming into the reasoning trace, we find that in distilled
> models, subtle differences in the opening sentence of the CoT can fully determine
> the model's refusal decision, and that these patterns transfer across models
> distilled from the same teacher. Finally, we extract linear refusal directions
> from model activations and show that ablating them increases harmful compliance,
> though less reliably than the same technique achieves on non-reasoning models, and
> with non-negligible degradation to general capabilities."

Two consequences: (a) for reasoning models, refusal is partly a *textual* event
inside the CoT, so a residual-stream direction is a coarser instrument; (b) a
distilled model's refusal decision can hinge on the *opening sentence* of the CoT
— a token-level, position-specific lever that a direction-level method will miss.

## 4. Latent Adversarial Training reorganizes the refusal representation

[arXiv:2504.18872](https://arxiv.org/abs/2504.18872), *Latent Adversarial Training
Improves the Representation of Refusal* (Llama 2 7B):

> "By computing activation differences between harmful and harmless instruction
> pairs and applying Singular Value Decomposition (SVD), we find that LAT
> significantly alters the refusal representation, concentrating it in the first two
> SVD components which explain approximately 75 percent of the activation
> differences variance — significantly higher than in reference models. This
> concentrated representation leads to more effective and transferable refusal
> vectors for ablation attacks: LAT models show improved robustness when attacked
> with vectors from reference models but become more vulnerable to
> self-generated vectors compared to SSFT and AT."

This is the "training changes the geometry" result. It is the direct rebuttal to
any claim that a refusal direction is a stable property of a model family: the
same training procedure that improves robustness also makes the direction easier
to find and easier to use against the model.

## 5. Sparse autoencoder view of refusal

Yeo et al. 2025, *Understanding Refusal in Language Models with Sparse
Autoencoders*. Findings of EMNLP 2025.
[arXiv:2505.23556](https://arxiv.org/abs/2505.23556) ·
[ACL](https://aclanthology.org/2025.findings-emnlp.338/)

> "Refusal is a key safety behavior in aligned language models, yet the internal
> mechanisms driving refusals remain opaque. In this work, we conduct a
> mechanistic study of refusal in instruction-tuned LLMs using sparse autoencoders
> to identify latent features that causally mediate refusal behaviors."

Reported findings: harmful-content features and refusal features are *distinct*;
harmful features can causally affect refusal features; jailbreaks can work by
suppressing refusal-related features. The refusal features also improve
generalization of linear probes to out-of-distribution adversarial samples.

Also relevant: the arXiv ID **2411.09003** (ACE) and **2505.23556** (SAE) are
two different papers; do not conflate the SAE refusal work with ACE.

## 6. Other useful results

- **Where the direction lives in layers.** Arditi's LessWrong reply: ablating
  layers before ~14 or after ~17 (Qwen 1.8B) has little effect; 14–17 alone
  suffices. Wollschläger: refusal is not a line but a cone. Zhong & Li: the
  persona-gating window is late-layer but model-specific.
- **Persona → refusal** ([arXiv:2606.26161](https://arxiv.org/abs/2606.26161)):
  compliant-persona steering drops Llama's refusal rate 97% → 2%.
- **COSMIC** (Siu et al., Findings of ACL 2025,
  [10.18653/v1/2025.findings-acl.1310](https://doi.org/10.18653/v1/2025.findings-acl.1310)):
  identifies refusal directions by cosine similarity *without* needing refusal
  token templates or output-text assumptions; works in adversarial settings and
  weakly-aligned models.
- **ACTOR** (Dabas et al., ICML 2025,
  [PMLR v267 dabas25a](https://proceedings.mlr.press/v267/dabas25a.html)):
  fine-tunes a *single layer* to reduce over-refusal, using a refusal vector to
  identify the activation components that trigger refusal.
- **RepE survey** (Bartoszcze et al. 2025,
  [arXiv:2502.17601](https://arxiv.org/abs/2502.17601)) and **RepControl survey**
  (ACM Computing Surveys, [10.1145/3846173](https://dl.acm.org/doi/10.1145/3846173)):
  the two broad surveys; both list performance degradation, compute overhead and
  steering precision as the open problems.

## 7. Datasets and metrics (practical table)

| Dataset | What it is | Use | Source |
| --- | --- | --- | --- |
| AdvBench `harmful_behaviors` | 520 harmful instructions | standard refusal benchmark | [arXiv:2307.15043](https://arxiv.org/abs/2307.15043), HF `walledai/AdvBench` |
| MaliciousInstruct | harmful instructions | Arditi's harmful set | [arXiv:2310.06987](https://arxiv.org/abs/2310.06987) |
| TDC2023 | harmful instructions | Arditi's harmful set | Mazeika et al. 2023 |
| HarmBench | 159 "standard behaviors" + classifier | ASR evaluation | [arXiv:2402.04249](https://arxiv.org/abs/2402.04249) |
| JailbreakBench | 100 harmful behaviors + 100 benign | refusal/safety score | [arXiv:2404.01318](https://arxiv.org/abs/2404.01318) |
| Alpaca | 52k harmless instructions | the "harmless" side of difference-in-means | Self-Instruct [arXiv:2212.10560](https://arxiv.org/abs/2212.10560), HF `tatsu-lab/alpaca` |
| XSTest | 250 safe prompts that *look* unsafe + 200 unsafe | over-refusal | [arXiv:2308.01263](https://arxiv.org/abs/2308.01263) |
| RMTBench | multi-turn user-centric roleplay, 80 characters, 8k+ dialogue rounds | over-refusal in roleplay | [arXiv:2507.20352](https://arxiv.org/abs/2507.20352) |
| OR-Bench | seemingly toxic but legitimate prompts | over-refusal | [arXiv:2405.20947](https://arxiv.org/abs/2405.20947) |
| PolyRefuse | harmful+benign prompts in 14 languages | cross-lingual refusal | [arXiv:2505.17306](https://arxiv.org/abs/2505.17306) |

**Metrics used in this literature:**

- *Refusal score* — substring match against a list of refusal phrases ("I'm
  sorry", "As an AI", …). Cheap, but Arditi et al. themselves note it "does not
  assess whether the completion is coherent or contains harmful content."
- *Safety score* — Llama Guard 2 as judge (Arditi et al.).
- *HarmBench ASR* — HarmBench's own classifier, with and without the model's
  default system prompt.
- *KL divergence from the original model* on harmless prompts — Heretic's
  collateral-damage measure, and the one the tooling community compares on.
- *Coherence* — judged separately from capability benchmarks, because a model
  can have intact MMLU and still write rambling prose.

**Minimum honest protocol** (assembled from the above): report refusal rate and
safety score on a harmful set, *plus* over-refusal on a benign set, *plus* at
least one capability benchmark, *plus* a coherence measure, *plus* KL from the
base model. Any single one of these alone is gameable — refusal rate alone is
exactly what covert non-compliance defeats.
