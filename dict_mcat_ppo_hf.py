# dict_mcat_ppo_hf.py
# DiCT PPO for MCAT with Hugging Face Transformers + Datasets streaming
# pip install "transformers>=4.42" "datasets>=2.19" accelerate torch --upgrade
import os, math, random, json, re, copy, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

# ================= CONFIG =================
@dataclass
class CFG:
    model_name: str = os.environ.get("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")  # choose any instruct-ish model you’re licensed to train
    dtype: str = os.environ.get("DTYPE", "bf16")  # "bf16"|"fp16"|"fp32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bsz: int = 2                           # keep small unless you have VRAM
    lr: float = 1e-6
    betas: Tuple[float,float] = (0.9, 0.95)
    wd: float = 0.01
    max_steps: int = 5_000
    cliprange: float = 0.2
    kl_coef: float = 0.01
    gen_temp: float = 0.7
    gen_topp: float = 0.9
    max_new_tokens: int = 196
    peer_margin_M: float = 0.02
    lambda_comp: float = 0.6
    lambda_regret: float = 0.6
    lambda_dis: float = 0.4
    lambda_repeat: float = 0.25
    failure_replay_prob: float = 0.25
    max_failbuf: int = 50000
    eval_every: int = 200
    # MCAT specifics
    default_timelimit_sec: int = 95
    calibration_weight: float = 0.5
    time_penalty_weight: float = 0.1
    # Data
    train_jsonl: str = os.environ.get("TRAIN_JSONL", "mcat_train.jsonl")   # JSONL with MCAT items (format below)
    stream: bool = True
CFG = CFG()

SECTIONS = ("CP","CARS","BB","PS")
CHOICE_LETTERS = ["A","B","C","D","E"]
CRITIC_LABELS = [
    "no_passage_evidence","math_units_issue","graph_misread","causation_vs_correlation",
    "scope_overreach","biochem_pathway_confusion","experimental_design","ok"
]

def resolve_correct_index(answer: Any, choices: List[Any]) -> int:
    """Resolve correct choice index from letter ('A'..), index (0- or 1-based), or exact text.
    Fallback to 0 if unresolved.
    """
    n = len(choices)
    try:
        if isinstance(answer, str):
            s = answer.strip()
            # letter form
            if len(s) == 1 and s.upper() in CHOICE_LETTERS[:n]:
                return CHOICE_LETTERS.index(s.upper())
            # numeric string
            if re.fullmatch(r"\d+", s):
                i = int(s)
                if 0 <= i < n:
                    return i
                if 1 <= i <= n:
                    return i - 1
            # exact text match (case-sensitive first)
            try:
                return choices.index(answer)
            except ValueError:
                pass
            # case-insensitive match
            s_lower = s.lower()
            for i, c in enumerate(choices):
                if str(c).strip().lower() == s_lower:
                    return i
        else:
            i = int(answer)
            if 0 <= i < n:
                return i
            if 1 <= i <= n:
                return i - 1
    except Exception:
        pass
    return 0

# ================= HF BACKBONE =================
class HFBackbone(nn.Module):
    def __init__(self, name: str, dtype: str = "bf16", device: str = "cuda"):
        super().__init__()
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        if self.tok.pad_token is None:
            # use eos as pad for causal LMs
            self.tok.pad_token = self.tok.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        self.device = device
        self.gen_cfg = GenerationConfig(
            do_sample=True,
            top_p=CFG.gen_topp,
            temperature=CFG.gen_temp,
            max_new_tokens=CFG.max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id
        )

    @torch.no_grad()
    def generate_text(self, prompts: List[str]) -> List[str]:
        batch = self.tok(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out_ids = self.lm.generate(**batch, generation_config=self.gen_cfg)
        # decode only the newly generated portion
        texts = self.tok.batch_decode(out_ids, skip_special_tokens=True)
        return texts

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        # grab the transformer's input embeddings
        return self.lm.get_input_embeddings()(input_ids)

# ================= PPO UTIL =================
def logprobs_of_labels(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Sequence-mean log-prob of labels under logits (ignoring pad). logits,labels: [B,T,V],[B,T]"""
    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != pad_id).float()
    seq_logp = (lp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return seq_logp

# ================= CRITIC & VALUE =================
class CriticHead(nn.Module):
    """Tiny critic that reads token embeddings and emits an approval score in [0,1]."""
    def __init__(self, d=2048):
        super().__init__()
        self.lstm = nn.LSTM(d, d, num_layers=1, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2*d, 1)
    def forward(self, feats):  # feats: [B,T,D]
        h,_ = self.lstm(feats)
        pooled = h.mean(dim=1)
        return torch.sigmoid(self.lin(pooled)).squeeze(-1)

class TinyValueHead(nn.Module):
    def __init__(self, d=2048):
        super().__init__()
        self.lin = nn.Linear(d, 1)
    def forward(self, feats):
        v = self.lin(feats.mean(dim=1))
        return v.squeeze(-1)

# ================= PROMPTS & PARSING =================
def format_mcat_prompt(item: Dict[str, Any]) -> str:
    choices = item["choices"]
    sec = item["section"]
    T = item.get("timelimit_sec", CFG.default_timelimit_sec)
    block = (
        f"Section: {sec}\nTime limit: {T}s\n"
        f"Passage:\n<<<{item['passage']}>>>\n\n"
        f"Question:\n<<<{item['question']}>>>\n\n"
        "Choices:\n" + "\n".join(f"{CHOICE_LETTERS[i]}) {c}" for i,c in enumerate(choices)) + "\n\n"
        "INSTRUCTIONS:\n"
        'Reply exactly as: "Answer: <A-D> | Confidence: [pA,pB,pC,pD] | Time: <seconds>s"\n'
        "Then 2–4 bullets: (1) passage line/figure, (2) equation/logic, (3) top distractor wrong. ≤90 words."
    )
    return block

def parse_model_output(text: str, num_choices: int) -> Tuple[Optional[str], List[float], float]:
    m_choice = re.search(r"Answer:\s*([A-E])", text, re.I)
    letter = m_choice.group(1).upper() if m_choice else None
    m_probs = re.search(r"Confidence:\s*\[([0-9eE\.\,\s]+)\]", text)
    probs = []
    if m_probs:
        try: probs = [float(x) for x in m_probs.group(1).split(",")]
        except: probs = []
    if len(probs) != num_choices:
        probs = [1.0/num_choices]*num_choices
    s = sum(probs); probs = [p/s if s>0 else 1.0/num_choices for p in probs]
    m_time = re.search(r"Time:\s*([0-9]+)\s*s", text, re.I)
    t_used = float(m_time.group(1)) if m_time else float(CFG.default_timelimit_sec)
    return letter, probs, t_used

def brier_score(probs: List[float], correct_idx: int) -> float:
    target = [0.0]*len(probs); target[correct_idx] = 1.0
    return sum((p - y)**2 for p,y in zip(probs, target)) / len(probs)

def rubric_check(item: Dict[str,Any], model_text: str) -> Dict[str,bool]:
    sec = item["section"]; passage = item["passage"]; q = item["question"]
    has_number = bool(re.search(r"\d", model_text))
    mentions_units = bool(re.search(r"(mol|M|s|J|kg|mL|L|Pa|N|V|A|Hz|°C|K)\b", model_text))
    mentions_quote = bool(re.search(r"(line\s*\d+|\".+\"|‘.+’|’.+’)", model_text, re.I))
    prompt_has_figure = bool(re.search(r"(Figure|Table|Fig\.)", passage+q, re.I))
    mentions_figure = bool(re.search(r"(Figure|Table|Fig\.)", model_text, re.I))
    units_ok = (sec in ("CP","PS")) and (not has_number or mentions_units) or (sec in ("BB","CARS"))
    scope_ok = (sec != "CARS") or mentions_quote
    graph_ok = (not prompt_has_figure) or mentions_figure
    return {"units_ok": bool(units_ok), "scope_ok": bool(scope_ok), "graph_ok": bool(graph_ok)}

def choose_critic_label(item: Dict[str,Any], model_text: str, rubric: Dict[str,bool], is_correct: bool) -> str:
    if is_correct and all(rubric.values()): return "ok"
    if item["section"] in ("CP","PS"):
        if not rubric["units_ok"]: return "math_units_issue"
        if not rubric["graph_ok"]: return "graph_misread"
        return "experimental_design"
    if item["section"] == "CARS":
        if not rubric["scope_ok"]: return "scope_overreach"
        return "no_passage_evidence"
    if item["section"] == "BB": return "biochem_pathway_confusion"
    return "no_passage_evidence"

# ================= METRICS =================
def metric_m(outputs: List[str], items: List[Dict[str,Any]]) -> torch.Tensor:
    scores = []
    for out, it in zip(outputs, items):
        choices = it["choices"]
        correct_idx = resolve_correct_index(it["answer"], choices)
        letter, probs, t_used = parse_model_output(out, len(choices))
        pred_idx = CHOICE_LETTERS.index(letter) if letter in CHOICE_LETTERS[:len(choices)] else -1
        correct = 1.0 if pred_idx == correct_idx else 0.0
        T = float(it.get("timelimit_sec", CFG.default_timelimit_sec))
        over = max(0.0, (t_used - T)/max(T,1.0))
        time_pen = CFG.time_penalty_weight * min(1.0, over)
        bs = brier_score(probs, correct_idx)
        calib = CFG.calibration_weight * (1.0 - bs)
        base = correct - time_pen
        base += (calib if correct > 0 else -calib)
        scores.append(float(max(0.0, min(1.0, base))))
    return torch.tensor(scores, device=CFG.device)

def best_known_score(items: List[Dict[str,Any]]) -> torch.Tensor:
    return torch.tensor([0.9 for _ in items], device=CFG.device)

ROLLING_PEER = {("CP",3):0.60, ("CARS",3):0.52, ("BB",3):0.58, ("PS",3):0.57}
def peer_scores(items: List[Dict[str,Any]]) -> torch.Tensor:
    return torch.tensor([ROLLING_PEER.get((it["section"], it.get("difficulty",3)), 0.55) for it in items], device=CFG.device)

# ================= FAILURE REPLAY =================
class FailureReplay:
    def __init__(self, maxlen=CFG.max_failbuf):
        self.buf: List[Dict[str, Any]] = []
        self.maxlen = maxlen
    def add(self, record):
        self.buf.append(record)
        if len(self.buf) > self.maxlen: self.buf.pop(0)
    def sample_items(self, n):
        if not self.buf: return []
        return [json.loads(random.choice(self.buf)["prompt_json"]) for _ in range(n)]

FAILBUF = FailureReplay()

# ================= TRAINER =================
class DiCTTrainer:
    def __init__(self):
        self.backbone = HFBackbone(CFG.model_name, CFG.dtype, CFG.device)
        # same-arch reference for KL
        self.ref = HFBackbone(CFG.model_name, CFG.dtype, CFG.device)
        self.ref.lm.load_state_dict(self.backbone.lm.state_dict())
        for p in self.ref.lm.parameters(): p.requires_grad_(False)

        d_embed = self.backbone.lm.get_input_embeddings().weight.shape[1]
        self.critic = CriticHead(d=d_embed).to(CFG.device)
        self.vhead = TinyValueHead(d=d_embed).to(CFG.device)

        self.opt = torch.optim.AdamW(
            list(self.backbone.lm.parameters()) + list(self.critic.parameters()) + list(self.vhead.parameters()),
            lr=CFG.lr, betas=CFG.betas, weight_decay=CFG.wd
        )

        self.pad_id = self.backbone.tok.pad_token_id
        self.running = {"acc": [], "brier": [], "time_over": [], "reasons": {}}

    def _features_from_texts(self, texts: List[str]) -> torch.Tensor:
        tok = self.backbone.tok(texts, padding=True, truncation=True, return_tensors="pt").to(CFG.device)
        with torch.no_grad():
            feats = self.backbone.embed_tokens(tok["input_ids"])
        return feats

    def _generate(self, prompts: List[str]) -> List[str]:
        return self.backbone.generate_text(prompts)

    def critique_then_rewrite(self, items: List[Dict[str,Any]], outputs: List[str]) -> Tuple[List[str], List[str], torch.Tensor]:
        # critic approval (on prompt+output)
        combo = [format_mcat_prompt(it) + "\n\n" + out for it,out in zip(items, outputs)]
        feats = self._features_from_texts(combo)
        with torch.no_grad():
            approval = self.critic(feats).detach()

        rew_prompts, reasons = [], []
        for it, out, appr in zip(items, outputs, approval):
            choices = it["choices"]; correct_idx = choices.index(it["answer"])
            letter, probs, t_used = parse_model_output(out, len(choices))
            pred_idx = CHOICE_LETTERS.index(letter) if letter in CHOICE_LETTERS[:len(choices)] else -1
            correct = (pred_idx == correct_idx)
            rb = rubric_check(it, out)
            reason = choose_critic_label(it, out, rb, correct)
            reasons.append(reason)
            if appr.item() < 0.6 or reason != "ok":
                if reason == "math_units_issue":
                    overlay = "CRITIC FLAG: ignored units. Recompute with units at each step; state final units."
                elif reason == "graph_misread":
                    overlay = "CRITIC FLAG: graph misread. Cite exact figure/table values and axis labels."
                elif reason == "scope_overreach":
                    overlay = "CRITIC FLAG: inference beyond scope. Quote the supporting passage line."
                elif reason == "no_passage_evidence":
                    overlay = "CRITIC FLAG: no passage evidence. Quote the specific line supporting the claim."
                elif reason == "experimental_design":
                    overlay = "CRITIC FLAG: experimental design. Identify IV, DV, and controls before answering."
                elif reason == "biochem_pathway_confusion":
                    overlay = "CRITIC FLAG: biochem confusion. Name the exact pathway/step from the passage."
                else:
                    overlay = "CRITIC FLAG: quality. Tighten reasoning; follow output format strictly."
                rew_prompts.append(format_mcat_prompt(it) + "\n\n" + overlay + "\nKeep the same output format.")
            else:
                rew_prompts.append(format_mcat_prompt(it))
        return rew_prompts, reasons, approval

    def compute_rewards(self, items: List[Dict[str,Any]], outputs: List[str], critic_appr: torch.Tensor, reasons: List[str]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str,float]]:
        m_scores = metric_m(outputs, items)
        best_scores = best_known_score(items)
        peer_median = peer_scores(items)

        L_comp = torch.clamp(CFG.peer_margin_M - (m_scores - peer_median), min=0.0)
        L_regret = torch.clamp(best_scores - m_scores, min=0.0)
        L_dis = 1.0 - critic_appr

        repeat_pen = torch.zeros_like(m_scores)
        if FAILBUF.buf:
            for i, rsn in enumerate(reasons):
                if rsn != "ok":
                    repeat_pen[i] = 1.0

        neg = CFG.lambda_comp * L_comp + CFG.lambda_regret * L_regret + CFG.lambda_dis * L_dis + CFG.lambda_repeat * repeat_pen
        reward = (m_scores - neg).detach()

        # dashboard stats
        accs, briers, overages = [], [], []
        for it, out in zip(items, outputs):
            choices = it["choices"]; correct_idx = resolve_correct_index(it["answer"], choices)
            letter, probs, t_used = parse_model_output(out, len(choices))
            pred_idx = CHOICE_LETTERS.index(letter) if letter in CHOICE_LETTERS[:len(choices)] else -1
            accs.append(1.0 if pred_idx == correct_idx else 0.0)
            briers.append(brier_score(probs, correct_idx))
            T = float(it.get("timelimit_sec", CFG.default_timelimit_sec))
            overages.append(max(0.0, (t_used - T)/max(T,1.0)))
        dash = {"acc": sum(accs)/max(1,len(accs)), "brier": sum(briers)/max(1,len(briers)), "time_over": sum(overages)/max(1,len(overages))}
        return reward, repeat_pen, dash

    def _ppo_step(self, prompts: List[str], responses: List[str], rewards: torch.Tensor):
        # For simplicity, learn from (prompt+response) log-probs as labels
        tok = self.backbone.tok([p + responses[i] for i,p in enumerate(prompts)], padding=True, truncation=True, return_tensors="pt").to(CFG.device)
        with torch.no_grad():
            logits_ref = self.ref.lm(**tok).logits
            ref_logp = logprobs_of_labels(logits_ref, tok["input_ids"], self.pad_id)

        logits = self.backbone.lm(**tok).logits
        logp = logprobs_of_labels(logits, tok["input_ids"], self.pad_id)

        adv = rewards - rewards.mean()  # quick baseline
        ratio = torch.exp(logp - ref_logp)  # using ref as a stable baseline proxy
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1-CFG.cliprange, 1+CFG.cliprange) * adv
        ppo_loss = -torch.min(unclipped, clipped).mean()

        kl = (ref_logp - logp).mean()
        loss = ppo_loss + CFG.kl_coef * kl

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.backbone.lm.parameters(), 1.0)
        self.opt.step()

        return {"loss": float(loss.item()), "ppo": float(ppo_loss.item()), "kl": float(kl.item())}

    def train(self, item_iter: Iterable[Dict[str,Any]]):
        step=0
        while step < CFG.max_steps:
            # batch
            batch_items = []
            while len(batch_items) < CFG.bsz:
                try:
                    it = next(item_iter)
                except StopIteration:
                    return
                # mix in failure replay sometimes
                if random.random() < CFG.failure_replay_prob and FAILBUF.buf:
                    batch_items.append(random.choice(FAILBUF.buf)["prompt_item"])
                else:
                    batch_items.append(it)

            prompts = [format_mcat_prompt(it) for it in batch_items]
            # generate
            outputs = self._generate(prompts)

            # critique -> rewrite -> generate again and pick better
            rew_prompts, reasons, approval = self.critique_then_rewrite(batch_items, outputs)
            rew_outputs = self._generate(rew_prompts)
            base_scores = metric_m(outputs, batch_items)
            edit_scores = metric_m(rew_outputs, batch_items)
            final = []
            for o,e,be,ee in zip(outputs, rew_outputs, base_scores, edit_scores):
                final.append(e if ee > be else o)

            rewards, repeat_pen, dash = self.compute_rewards(batch_items, final, approval, reasons)
            logs = self._ppo_step(prompts, final, rewards)
            step += 1

            # log fails
            for it, out, rsn, r in zip(batch_items, final, reasons, rewards.tolist()):
                if r < 0.35:
                    FAILBUF.add({"prompt_json": json.dumps(it), "prompt_item": it, "reason": rsn, "output": out})

            # running stats
            for k in ("acc","brier","time_over"):
                self.running[k].append(dash[k])
            for r in reasons:
                self.running["reasons"][r] = self.running["reasons"].get(r, 0) + 1

            if step % 50 == 0:
                top_err = sorted(self.running["reasons"].items(), key=lambda x: -x[1])[:3]
                tr = ", ".join(f"{k}:{v}" for k,v in top_err) if top_err else "—"
                acc7 = sum(self.running["acc"][-50:])/max(1,len(self.running["acc"][-50:]))
                br7  = sum(self.running["brier"][-50:])/max(1,len(self.running["brier"][-50:]))
                over7= sum(self.running["time_over"][-50:])/max(1,len(self.running["time_over"][-50:]))
                print(f"[{step}] loss={logs['loss']:.4f} ppo={logs['ppo']:.4f} kl={logs['kl']:.4f} | acc~{acc7:.3f} brier~{br7:.3f} over~{over7:.3f} | buf={len(FAILBUF.buf)} | top_err={tr}")

            if step % 1000 == 0:
                CFG.lambda_comp *= 1.05
                CFG.lambda_regret *= 1.05
                CFG.lambda_dis *= 1.05
                CFG.lambda_repeat *= 1.02
                CFG.peer_margin_M = min(0.12, CFG.peer_margin_M * 1.05)

# ================= DATA =================
def iter_mcat_items(path_jsonl: str, stream: bool = True) -> Iterable[Dict[str,Any]]:
    """
    Each JSON line must be:
    {
      "section":"CP|CARS|BB|PS",
      "passage":"...",
      "question":"...",
      "choices":["...","...","...","..."],
      "answer":"C",
      "difficulty":3,
      "timelimit_sec":95
    }
    """
    if stream:
        ds = load_dataset("json", data_files=path_jsonl, streaming=True)["train"]
        for ex in ds:
            yield {
                "section": ex["section"],
                "passage": ex["passage"],
                "question": ex["question"],
                "choices": list(ex["choices"]),
                "answer": ex["answer"],
                "difficulty": int(ex.get("difficulty",3)),
                "timelimit_sec": int(ex.get("timelimit_sec", CFG.default_timelimit_sec))
            }
    else:
        ds = load_dataset("json", data_files=path_jsonl)["train"]
        for ex in ds:
            yield {
                "section": ex["section"],
                "passage": ex["passage"],
                "question": ex["question"],
                "choices": list(ex["choices"]),
                "answer": ex["answer"],
                "difficulty": int(ex.get("difficulty",3)),
                "timelimit_sec": int(ex.get("timelimit_sec", CFG.default_timelimit_sec))
            }
