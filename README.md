# LLM Fine-Tuning with GRPO: Practical Examples

This repository contains practical, experiment-driven examples of fine-tuning large language models using **reinforcement learning**, with a particular focus on **Group Relative Policy Optimisation (GRPO)**.

The goal of this project is to explore how reinforcement learning techniques—most notably GRPO—can be applied effectively to **realistic, industry-relevant use cases**, such as information extraction, using modest compute and simple reward signals.

---

## Motivation

Most reinforcement learning research for LLMs today focuses on:

* mathematical reasoning benchmarks,
* step-by-step problem solving,
* large-scale, research-oriented setups.

While valuable, these benchmarks do not reflect how LLMs are commonly used in production environments.

In industry, LLMs are often embedded in pipelines where:

* outputs must be **accurate and consistent**, not impressive,
* behaviour matters more than verbosity,
* downstream validation always exists,
* improvement is incremental rather than absolute.

This repository explores how GRPO-style training can help shape **model behaviour** for such tasks.

---

## What this repository contains

* **End-to-end GRPO fine-tuning examples**

  * Prompt design
  * Dataset preprocessing
  * Reward function design
  * Training configuration
* **Information extraction use cases**

  * Named Entity Recognition (NER) as a proxy for real-world extraction tasks
  * Rule-based and relative reward signals
* **Experimentation-focused workflows**

  * Iterative reward design
  * Prompt refinement
  * Batch size and training length trade-offs

The examples are intentionally small, explicit, and easy to reason about.

---

## Current example: GRPO for Named Entity Recognition

The initial example demonstrates how to apply GRPO to a **Named Entity Recognition** task using the CoNLL-2003 dataset.

Key aspects include:

* Framing NER as a conversational extraction task
* Enforcing strict output structure via rewards
* Using relative and rule-based signals rather than relying purely on token accuracy
* Prioritising structural correctness and consistency early in training

This setup mirrors common information extraction problems found in domains such as finance and insurance, where extracted entities feed downstream systems rather than end users.


## Disclaimer

This project is exploratory by design.
It does not claim to provide optimal or production-ready training recipes.

All examples are intended to:

* illustrate concepts,
* encourage experimentation,
* and provide a practical starting point for applying reinforcement learning to LLM fine-tuning.

