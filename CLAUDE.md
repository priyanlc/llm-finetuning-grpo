# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Experiment-driven repository exploring **GRPO (Group Relative Policy Optimization)** for fine-tuning LLMs on practical information extraction tasks. Uses Named Entity Recognition (BIO tagging on CoNLL-2003) as the target task with Qwen/Qwen2.5-3B-Instruct as the base model.

This is a research/exploration project — not a production system. All code lives in Jupyter notebooks.

## Repository Structure

- `IL_CoNLL_grpo_IL_v0.1.ipynb` — First GRPO experiment
- `IL_CoNLL_grpo_IL_v0.2.ipynb` — Latest GRPO experiment (625 training steps, improved reward weighting)
- `IL_CoNLL_grpo_IL_v0.1.html` — Rendered output of v0.1

No separate Python modules, packages, or build system. Dependencies are installed inline in notebooks via `pip install`.

## Key Dependencies

`trl`, `peft`, `datasets`, `transformers`, `torch`, `wandb`, `weave`, `huggingface_hub`

## Architecture

### Training Pipeline

1. Parse CoNLL-2003 data → conversation-formatted prompts (system + user messages)
2. Load Qwen2.5-3B-Instruct with LoRA adapters (r=32, alpha=64, ~59.8M trainable params / 1.9%)
3. Pre-training evaluation on test set
4. GRPO training via TRL's `GRPOTrainer`
5. Post-training evaluation
6. Merge LoRA weights → push to Hugging Face Hub

### Reward System (5 signals, weighted)

The combined reward formula: `0.2*format + 3.0*alignment + 3.0*bio_validity + 0.1*accuracy + 0.1*reasoning`

| Function | What it checks | Returns |
|---|---|---|
| `reward_format` | `<reasoning>` and `<answer>` sections present | 1.0 / -1.0 |
| `reward_token_alignment` | Token count matches gold (no missing/extra/reordered) | 0.0–1.0 |
| `reward_bio_validity` | BIO sequence rules (B-TAG before I-TAG, etc.) | 1.0 / -1.0 |
| `reward_token_accuracy` | Per-token label correctness vs gold | 0.0–1.0 |
| `reward_reasoning_presence` | Non-empty reasoning section | 0.5 / 0.0 |

Heavy weighting on alignment and BIO validity encourages structural correctness first.

### GPU Memory Considerations

Training config parameters (batch size, gradient accumulation steps) are tuned per GPU tier. Comments in notebooks indicate settings for 24GB vs 80GB+ GPUs.

## Experiment Tracking

- **Weights & Biases** (`wandb`) for metrics logging
- **Weave** for tracing
- Custom `evaluate_model_on_conll` function computes per-reward metrics and prints sample outputs
