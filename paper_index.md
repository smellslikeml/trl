# Paper Index

A list of algorithms, methods, and training approaches implemented in TRL that originate
from research papers, with a link to the paper and a short description of what is implemented.

## EDGE-GRPO: Entropy-Driven GRPO with Guided Error Correction for Advantage Diversity

[Paper](https://huggingface.co/papers/2507.21848)

Addresses GRPO's *advantage collapse* problem (identical rewards within a group yield zero
advantage and no gradient) by introducing two mechanisms. This repo implements the
**Entropy-Driven Advantage (EDA)** signal-level contribution: each response's advantage is
divided by its group-normalized policy entropy (`P̂_i = P_i / mean_group(P)`,
`Â_i = A_i / P̂_i`), so confident-correct responses receive a larger advantage and
confident-wrong ones a harsher penalty, increasing advantage diversity.

It is available in the [`GRPOTrainer`] via the `entropy_driven_advantage` config option,
which applies the reweighting at advantage-computation time using the generation-time policy
entropy.

The paper's response-level **Guided Error Correction (GEC)** (regenerate / answer-injection /
reference-solution replacement for incorrect completions) is out of scope here: it requires
per-question external reference solutions, a dataset-specific data pipeline that does not fit
the generic trainer.
