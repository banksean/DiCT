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
pip install bitsandbytes  # for 4-bit loading on limited VRAM
```

---

## Dataset Format

The trainer expects a JSONL file where each line is an MCAT item:

```
{
  "section": "CP",  // one of CP, CARS, BB, PS
  "passage": "Passage text here...",
  "question": "Question text here...",
  "choices": ["Choice A", "Choice B", "Choice C", "Choice D"],
  "answer": "C",           // correct choice letter
  "difficulty": 3,         // optional int 1–5
  "timelimit_sec": 95      // optional per-question time limit
}
```

---

## Usage

1. Prepare your dataset
Place your MCAT practice set in mcat_train.jsonl (or update CFG.train_jsonl in the script).

1. Choose your base model
Set the environment variable:

```bash
export MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"
```

You can substitute any causal LM that fits your VRAM budget.

3. Run training

```bash
python dict_mcat_ppo_hf.py
```

---

## Training Loop Overview

1. Sample a batch of MCAT items (mix in failure replay items).
1. Prompt the model to answer with:
```yaml
Answer: <A-D> | Confidence: [pA,pB,pC,pD] | Time: <seconds>s
```
plus a short reasoning section.
1. Score the answer:
  - Correctness (+)
  - Timing penalty (−)
  - Calibration bonus/penalty
1. Compare to peer median and regret baseline.
1. Critique with the disapproval critic and label the mistake.
1. Rewrite the answer with targeted feedback.
1. Pick the better answer (bandit selection).
1. Update the policy via PPO with all “shame” terms.
1. Log top mistake types and rolling accuracy/Brier/time-over.

---

## Failure Replay
The trainer stores:
- Prompt JSON
- Mistake type
- Wrong output

These are periodically re-injected into training to focus on weaknesses.

---

## Extending the Critic
The critic is currently a lightweight LSTM head over token embeddings.
You can:
- Train it on human-labeled MCAT mistakes.
- Replace it with a separate preference/rubric model.
- Expand the taxonomy of mistake types.

---

## License

This repo is provided as-is for research and educational purposes.
Check the license terms of any base model you use before fine-tuning or deployment.

---

## Citation
If you use this method in academic work, please cite the concept as:

> Disapproval-Guided Comparative Training for Domain-Specific Mastery (DiCT-PPO), 2025.

---

## 4-bit training on a single 12 GB GPU plug-and-play

````markdown
## Run on a 12 GB GPU (4-bit)

If you’ve got ~12 GB of VRAM (e.g., RTX 3060), run the model in **4-bit** with `bitsandbytes`.

### 1) Install deps
```bash
pip install bitsandbytes "transformers>=4.42" "datasets>=2.19" accelerate torch --upgrade
````

### 2) (One time) Configure Accelerate

```bash
accelerate config
# Choose: single-GPU, no DeepSpeed, BF16 if supported, otherwise FP16.
```

### 3) Enable 4-bit in the script

Open `dict_mcat_ppo_hf.py` and change the HF model load to include 4-bit. Replace the `AutoModelForCausalLM.from_pretrained(...)` call with this:

```python
from transformers import BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if CFG.dtype == "bf16" else torch.float16,
)

self.lm = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16 if CFG.dtype == "bf16" else torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    quantization_config=bnb_cfg,
    device_map="auto",             # lets HF place layers across your GPU/CPU if needed
)
```

> Tip: keep `self.tok.pad_token = self.tok.eos_token` for causal LMs.

### 4) Train

```bash
export MODEL_NAME="microsoft/Phi-3-mini-4k-instruct"   # or another small instruct model
export TRAIN_JSONL="mcat_train.jsonl"
accelerate launch dict_mcat_ppo_hf.py
```

**Expected VRAM:** \~8–12 GB depending on model size and context length.
If you OOM:

* lower `CFG.max_new_tokens` (e.g., 128),
* reduce `CFG.bsz` to 1,
* or try a smaller base model (e.g., `Qwen2.5-0.5B-Instruct`).

### Optional: LoRA fine-tuning (PEFT)

For even lower memory, layer-freeze the base model and train small **LoRA** adapters. That’s a \~5–30 MB memory delta and plays nicely with 4-bit. If you want, I can add a PEFT/LoRA variant of the script next.

```
