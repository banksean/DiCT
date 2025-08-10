# DiCT-PPO-MCAT

**Disapproval-Guided Comparative Training** (DiCT) with **Proximal Policy Optimization** (PPO) for MCAT mastery.

This repo implements a reinforcement learning loop that trains a language model to *study for and achieve a top score on the MCAT*.  
The core idea:  
- Give the model MCAT questions and grade them.  
- Penalize it comparatively when peers/previous checkpoints do better.  
- Penalize it for "regrettable" mistakes where an ideal answer is known.  
- Penalize it for failing a **rubric-based disapproval critic** (units, scope, graph reading, etc.).  
- Penalize it for **repeating** past mistake types.  
- Give it a chance to rewrite answers after critique.  
- Iterate until the model stops making those mistakes.

---

## Features

- **Hugging Face Transformers backbone**  
  Loads any causal language model from the Hub (`microsoft/Phi-3-mini-4k-instruct`, `Qwen2.5-1.5B-Instruct`, etc.).
- **MCAT-specific reward shaping**  
  - Correctness, timing penalty, and confidence calibration (Brier score).  
  - Regret loss vs. a best-known rubric baseline.  
  - Comparative loss vs. rolling peer median.
- **Disapproval critic**  
  - Trained to detect MCAT-specific mistake types:
    - `math_units_issue`
    - `graph_misread`
    - `scope_overreach`
    - `no_passage_evidence`
    - `experimental_design`
    - `biochem_pathway_confusion`
    - etc.
- **Failure replay buffer**  
  - Stores past wrong answers with mistake labels.  
  - Oversamples recurring errors until they’re fixed.
- **Critique → Rewrite loop**  
  - Model receives targeted feedback on why its first answer was wrong.  
  - Encouraged to self-correct before the final graded attempt.
- **Streaming dataset support**  
  - Train directly on a JSONL file of MCAT items via `datasets` streaming mode.

---

## Requirements

```bash
pip install "transformers>=4.42" "datasets>=2.19" accelerate torch
