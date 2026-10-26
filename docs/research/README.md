---
title: Refusal research notes — index
tags: [research, refusal, steering, interpretability, safety]
---

# Research notes: refusal, abliteration, steering

Reading notes for the neuron-explorer project. Every claim below is backed by a
primary source (arXiv / paper) that was fetched while writing; the quote and the
link are given inline.

| Note | Covers |
| --- | --- |
| [refusal-direction-and-abliteration.md](refusal-direction-and-abliteration.md) | Arditi et al. 2024 (the "single direction" result), the abliteration recipe, and the multi-direction critiques that follow it. |
| [steering-and-persona-vectors.md](steering-and-persona-vectors.md) | Activation steering as a family: ActAdd/CAA, RepE, Persona Vectors, ACE, and how persona gates refusal. |
| [refusal-safety-and-robustness.md](refusal-safety-and-robustness.md) | Defenses, attacks, reasoning models, SAE views, and the evaluation datasets used across this literature. |
| [implications-for-neuron-explorer.md](implications-for-neuron-explorer.md) | What the explorer's current machinery can and cannot express, and the research questions that fall out. |

## Repo issue check (2026-10-26)

Checked the fork's issue tracker for anything already covering this research.
There is **no** `research` label and no ticket about refusal, abliteration or
steering theory. The relevant surface is the wayfinder map
([#1](https://github.com/amaldevice/neuron-explorer/issues/1)) and its
`wayfinder:research` children:

| Issue | Relation to this research |
| --- | --- |
| #2 Can the steering sweep run on Apple Silicon? | The compute prerequisite — a refusal-direction experiment needs only forward passes, so this path is enough. |
| #3 Does the steering fork support LFM2 and Granite-4.0? | Same, for the subject models a refusal experiment would use. |
| #4 Subject-model adapter facts | Relevant: difference-in-means needs the *right* residual-stream read point and norm convention per architecture. |
| #5 / #6 omp and pi agent backends | Not directly related. |
| #13 Which sweep path does the spec endorse for later? | Tangential — a sweep is not needed for direction extraction. |

Labels present on the fork: `needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, `wontfix` (triage vocabulary), plus `wayfinder:map`,
`wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`,
`wayfinder:task`. There is no plain `research` label — the research label is
`wayfinder:research`.

Two operational notes, both contradicting `docs/agents/issue-tracker.md`:
`gh` from a plain clone resolves to **upstream** (`lyramakesmusic/neuron-explorer`),
so every command must pass `-R amaldevice/neuron-explorer`; and the upstream
repo has no issues at all.

## Scope and honesty notes

- I did **not** run any model. Everything here is read from papers and the
  abliteration community wiki, not measured locally.
- The abliteration community wiki (`abliteration.org`) is a practitioner
  taxonomy (M1–M8 method catalog), not peer-reviewed. I use it for tooling
  facts and cross-links, and label it as such; every scientific claim in these
  notes is attributed to a paper.
- Some sources in the notes are dated 2026 (e.g. `arXiv:2606.26161`,
  `arXiv:2602.02132`). Every arXiv ID in these notes was resolved by fetching
  its `arxiv.org/abs/` page and checking the title, but the venue attribution
  for the 2026 ones comes from the paper's own abstract page, not from an
  independent proceedings listing.
