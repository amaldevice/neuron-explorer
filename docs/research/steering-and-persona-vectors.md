---
title: "Steering and persona vectors — ActAdd, RepE, Persona Vectors, and persona-gated refusal"
tags: [research, steering, persona, refusal, interpretability]
---

# Steering and persona vectors

The second requested paper: Chen et al. (Anthropic), *Persona Vectors: Monitoring
and Controlling Character Traits in Language Models.*
[arXiv:2507.21509](https://arxiv.org/abs/2507.21509) ·
[code](https://github.com/safety-research/persona_vectors)

## 1. Why persona is a separate axis from refusal

The persona-vectors paper is not about refusal. It is about character traits, and
its relevance here is that it establishes the *method* that a steering-first tool
wants, and that it exposes a trait axis (persona) that turns out to gate refusal.

> "Large language models interact with users through a simulated 'Assistant'
> persona. While the Assistant is typically trained to be helpful, harmless, and
> honest, it sometimes deviates from these ideals. In this paper, we identify
> directions in the model's activation space—persona vectors—underlying several
> traits, such as evil, sycophancy, and propensity to hallucinate."

## 2. The extraction pipeline

Given only a natural-language description of a trait, the pipeline is fully
automatic:

1. **Contrastive generation.** For each question in an extraction set, generate
   responses under a *positive* system prompt (elicits the trait) and a *negative*
   system prompt (suppresses it), 10 rollouts each.
2. **Score filtering.** Keep only responses that match the intent of their prompt:
   trait score $> 50$ for positive prompts, $< 50$ for negative prompts.
3. **Activation capture.** Extract residual stream activations at every layer,
   **averaged across response tokens**. (The paper notes response tokens give
   better steering directions than prompt tokens — Appendix A.3.)
4. **Difference in mean activations** between trait-expressing and
   non-trait-expressing responses → one candidate persona vector per layer.
5. **Layer selection** by testing steering effectiveness across layers
   (Appendix B.4).

The vector for layer $\ell$ is then used two ways:

**Steering (control):**

$$
h_\ell \leftarrow h_\ell + \alpha \cdot v_\ell
$$

at each decoding step, where $\alpha$ is a scalar steering coefficient.
Reported effects: steering toward *evil* produces violent content, toward
*sycophancy* produces excessive agreement and flattery, toward *hallucination*
produces elaborate fabrications.

**Monitoring (detection):** project the activation at the *final prompt token*
(the token right before the Assistant's response) onto the persona vector. Over
system prompts that interpolate from trait-discouraging to trait-promoting, the
projection correlates with trait expression in the following response at
$r = 0.75\text{--}0.83$. The paper is careful about the ceiling: the correlation
"arise[s] primarily from distinguishing between different prompt types … with
more modest correlations when controlling for prompt type." So persona vectors
are good at detecting *explicit* prompt-induced shifts and less good at subtle
deployment-time drift.

Models: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Traits: evil,
sycophancy, hallucination in the main text; optimism, humor and others in
Appendix G.

## 3. Persona shifts during finetuning

Three findings here are directly reusable:

- **Persona shifts after finetuning are predictable from activations.** Both
  intended and *unintended* personality changes correlate strongly with shifts
  along the relevant persona vectors. Unintended examples: training on
  "EM-like" datasets (flawed medical advice, invalid math solutions, insecure
  code) induces persona changes with no matching trait in the data — e.g.
  training on flawed math reasoning increases expression of *evil*.
- **Post-hoc intervention mitigates the shift.** Steering back along the persona
  vector after finetuning reduces the shift.
- **Preventative steering avoids it.** During finetuning, actively steer the model
  *toward* the undesired persona direction, "relieving the model of the need to
  shift in that direction to fit the training data" — the objective's pressure is
  cancelled out. Coherence stays above 80 on average; MMLU is preserved better
  than inference-time steering; multi-layer steering is stronger still.
- **A regularizer on the projection does not work.** Penalizing changes in the
  projection along the trait direction during training "is ineffective in
  practice." The paper's hypothesis: optimization pressure pushes the model to
  represent the trait using *alternative* directions. This is a genuinely
  useful negative result for anyone planning a representation-level regularizer.
- Compared against **CAFT** (Casademunt et al. 2025), which zero-ablates concept
  directions during training: CAFT works for evil and sycophancy but not for
  hallucination.

## 4. Persona gates refusal

*Refusal Lives Downstream of Persona in Chat Models* (Zhong & Li).
[arXiv:2606.26161](https://arxiv.org/abs/2606.26161) · ICML 2026
Mechanistic Interpretability Workshop.

> "We show they interact: a compliant persona gates refusal. In Qwen2.5-7B-Instruct
> and Llama-3.1-8B-Instruct, we extract a compliant model-persona direction and a
> refusal direction and intervene on both. Compliant persona steering suppresses
> refusal — in Llama, the refusal rate falls from 97% to 2%. Reintroducing the
> refusal direction partially restores refusal at late layers but not at early
> ones. Projecting out the persona direction in a late-layer window restores it to
> baseline; projecting out a random direction does not. Refusal is therefore gated
> at the late-layer expression stage, downstream of where it is computed."

This is the single most important result for a tool that manipulates directions:
**refusal is not just a direction you can remove — it is a signal that a persona
direction gates on the way out.** A steering tool that ablates a refusal
direction but leaves the compliant-persona direction intact is fighting a
one-sided battle; and conversely, a "compliant persona" steering vector may be a
cheaper refusal-removal lever than the refusal vector itself.

Their own limitations: two 7–8B instruct models only; the mediating window is
model-specific (L20 in Llama, L20–L22 in Qwen), so the locus generalizes as a
*late-layer effect*, not a fixed layer; scoring relies on model-based judges.

## 5. The steering lineage these papers sit in

| Work | Contribution | Link |
| --- | --- | --- |
| Turner et al. 2023, *Activation Addition* | ActAdd: inference-time control by adding activation differences from paired prompts, preserving off-target performance | [arXiv:2308.10248](https://arxiv.org/abs/2308.10248) |
| Rimsky et al. 2023, *Steering Llama 2 via Contrastive Activation Addition* | CAA: averages residual-stream differences between positive/negative pairs of a behavior; direct precursor to difference-in-means | [arXiv:2312.06681](https://arxiv.org/abs/2312.06681) |
| Zou et al. 2023, *Representation Engineering* | RepE framing: concept vectors as first-class read/control objects; also the "harmfulness" vs "refusal" distinction | [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) |
| Stolfo et al. 2024, *InstructSteer* | Instruction-specific steering vectors; compositionality; transfer from instruct-tuned to base models | [arXiv:2410.12877](https://arxiv.org/abs/2410.12877) |
| Marshall et al. 2024, *Refusal in LLMs is an Affine Function* | ACE: affine subspace projection + addition; generalizes earlier methods | [arXiv:2411.09003](https://arxiv.org/abs/2411.09003) |
| Bartoszcze et al. 2025 | RepE survey: taxonomy, risks (performance drop, compute, steerability), agenda | [arXiv:2502.17601](https://arxiv.org/abs/2502.17601) |
| Chen et al. 2025, *Persona Vectors* | Automated trait-direction extraction from a natural-language trait description | [arXiv:2507.21509](https://arxiv.org/abs/2507.21509) |

Two RepE findings worth carrying forward, because they bear on any claim that a
direction "means" something:

- **Refusal ≠ harmfulness as a representation.** RepE and Zheng et al. both find
  a *separate* harmfulness direction that is not altered by adversarial suffixes.
  Arditi et al. found the opposite for their refusal direction (§5 of their
  paper explicitly contrasts with Zou et al.). The disagreement is live, and
  later work sides with the separation view: *LLMs Encode Harmfulness and
  Refusal Separately* ([arXiv:2507.11878](https://arxiv.org/abs/2507.11878))
  identifies a harmfulness direction distinct from the refusal direction, with
  the causal asymmetry that steering along harmfulness makes a model treat
  harmless instructions as harmful, while steering along refusal elicits
  refusal directly without the harmfulness reinterpretation.
- **The harmfulness concept survives jailbreaks** in RepE's Vicuna-13B
  experiments: >90% classification accuracy of harmful vs harmless holds even
  under a manual jailbreak or a GCG suffix. So the model may *perceive* the
  request as harmful while still complying — a gap between detection and
  expression, which is the same shape as the persona-gating result above.

## 6. What a steering tool must therefore be able to express

Collecting the requirements from the papers above:

1. **Add a direction at a layer, all token positions** (persona steering, refusal
   induction).
2. **Project out a direction at all layers, all token positions** (directional
   ablation).
3. **Project out a direction at a *subset* of layers** — the "ablate a mid-to-late
   band" practice, and the persona-gating window finding.
4. **Project out a *set* of directions** (multi-direction / cone / RFM-AGOP).
5. **Ablate a direction *at inference* vs. bake it into the weights** — the
   equivalence, and the difference in reproducibility.
6. **Steer *during* training** (preventative steering) — a train-time hook, not
   just an inference hook.
7. **Read the projection of a hidden state onto a direction** (monitoring) —
   including at the *final prompt token* specifically, which is where the
   persona-vectors paper gets its best correlations.
8. **Ablate at a *feature* level, not a direction level** (SAE refusal features,
   §6 of the sibling note) — i.e. hook a neuron's activation, not the residual
   stream.
