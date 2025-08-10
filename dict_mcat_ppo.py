# dict_mcat_ppo.py
# Disapproval-Guided Comparative Training (DiCT) with PPO for MCAT
# pip install torch torchmetrics transformers accelerate datasets
import math, random, json, time, re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= CONFIG =================
@dataclass
class CFG:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bsz: int = 8
    grad_accum: int = 1
    lr: float = 1e-5
    betas: Tuple[float,float] = (0.9, 0.98)
    wd: float = 0.01
    max_steps: int = 10_000
    cliprange: float = 0.2
    kl_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    gen_topk: int = 0
    gen_topp: float = 0.9
    gen_temp: float = 0.7
    n_candidates: int = 2
    peer_margin_M: float = 0.02
    lambda_comp: float = 0.6
    lambda_regret: float = 0.6
    lambda_dis: float = 0.4
    lambda_repeat: float = 0.25
    failure_replay_prob: float = 0.25
    max_failbuf: int = 20000
    eval_every: int = 500
    max_len: int = 512
    # MCAT specifics
    default_timelimit_sec: int = 95
    calibration_weight: float = 0.5   # how much (1-Brier) affects reward
    time_penalty_weight: float = 0.1  # max penalty when doubling timelimit
CFG = CFG()

SECTIONS = ("CP","CARS","BB","PS")
CRITIC_LABELS = [
    "no_passage_evidence",
    "math_units_issue",
    "graph_misread",
    "causation_vs_correlation",
    "scope_overreach",
    "biochem_pathway_confusion",
    "experimental_design",
    "ok"
]

# ================= MODELS (placeholders) =================
class PolicyModel(nn.Module):
    """
    Replace with a real transformers LM. This is a stub so the file runs.
    """
    def __init__(self, vocab_size=32000, d=768):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.lm = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d, nhead=12, dim_feedforward=4*d, batch_first=True),
            num_layers=6
        )
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, x, attn=None):
        h = self.tok_emb(x)
        h = self.ln(h)
        h = self.lm(h, h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_len=CFG.max_len, temperature=CFG.gen_temp, top_p=CFG.gen_topp):
        bsz, t = prompt_ids.shape
        ids = prompt_ids.clone()
        for _ in range(max_len - t):
            logits = self.forward(ids)[:, -1, :]
            if temperature > 0:
                logits = logits / max(1e-5, temperature)
                probs = torch.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    cut_mask = (cumsum <= top_p).float()
                    keep = torch.zeros_like(probs).scatter(1, sorted_idx, cut_mask)
                    probs = probs * keep
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                next_id = torch.multinomial(probs, 1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

class CriticHead(nn.Module):
    """
    Disapproval critic: returns approval in [0,1].
    In practice train on redlined rationales; here it’s a lightweight stub.
    """
    def __init__(self, d=768):
        super().__init__()
        self.encoder = nn.LSTM(input_size=d, hidden_size=d, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2*d, 1)

    def forward(self, feats):  # feats: [B,T,D]
        h, _ = self.encoder(feats)
        pooled = h.mean(dim=1)
        score = torch.sigmoid(self.lin(pooled))
        return score  # ≈ approval probability

# ================= TOKENIZER / DATA (stubs) =================
class DummyTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    def encode(self, s): return [self.bos_id] + [min(31999, 10+ord(c)%100) for c in s][:300] + [self.eos_id]
    def decode(self, ids): return "".join(chr((i-10)%100 + 27) for i in ids)

TOK = DummyTokenizer()

def collate_text(batch: List[str]) -> torch.Tensor:
    toks = [TOK.encode(s) for s in batch]
    mlen = max(len(t) for t in toks)
    out = torch.full((len(toks), mlen), TOK.pad_id, dtype=torch.long)
    for i,t in enumerate(toks):
        out[i,:len(t)] = torch.tensor(t, dtype=torch.long)
    return out.to(CFG.device)

# ==================== MCAT UTILITIES ====================
CHOICE_LETTERS = ["A","B","C","D","E"]

def resolve_correct_index(answer: Any, choices: List[Any]) -> int:
    """Resolve index from letter ('A'..), index (0/1-based), or exact text; fallback 0."""
    n = len(choices)
    try:
        if isinstance(answer, str):
            s = answer.strip()
            if len(s) == 1 and s.upper() in CHOICE_LETTERS[:n]:
                return CHOICE_LETTERS.index(s.upper())
            if re.fullmatch(r"\d+", s):
                i = int(s)
                if 0 <= i < n: return i
                if 1 <= i <= n: return i - 1
            try:
                return choices.index(answer)
            except ValueError:
                s_lower = s.lower()
                for i,c in enumerate(choices):
                    if str(c).strip().lower() == s_lower:
                        return i
        else:
            i = int(answer)
            if 0 <= i < n: return i
            if 1 <= i <= n: return i - 1
    except Exception:
        pass
    return 0

def format_mcat_prompt(item: Dict[str, Any]) -> str:
    sec = item["section"]
    T = item.get("timelimit_sec", CFG.default_timelimit_sec)
    choices = item["choices"]
    prompt = (
        f"Section: {sec}\nTime limit: {T}s\n"
        f"Passage:\n<<<{item['passage']}>>>\n\n"
        f"Question:\n<<<{item['question']}>>>\n\n"
        f"Choices:\n" + "\n".join(f"{CHOICE_LETTERS[i]}) {c}" for i,c in enumerate(choices)) + "\n\n"
        "INSTRUCTIONS:\n"
        '- Reply exactly as:\n'
        '"Answer: <A-D> | Confidence: [pA,pB,pC,pD]"\n'
        "- Then 2–4 bullets: (1) passage line/figure cited, (2) equation/logic, (3) top distractor wrong.\n"
        "- Keep under 90 words."
    )
    return prompt

def parse_model_output(text: str, num_choices: int) -> Tuple[Optional[str], List[float], float]:
    """
    Returns (choice_letter, probs[0..n-1], time_used_sec)
    We expect 'Answer: C | Confidence: [0.1,0.2,0.6,0.1] | Time: 72s' (time optional).
    """
    # choice
    m_choice = re.search(r"Answer:\s*([A-E])", text, re.I)
    letter = m_choice.group(1).upper() if m_choice else None
    # probs
    m_probs = re.search(r"Confidence:\s*\[([0-9eE\.\,\s]+)\]", text)
    probs = []
    if m_probs:
        try:
            probs = [float(x) for x in m_probs.group(1).split(",")]
        except Exception:
            probs = []
    if len(probs) != num_choices:
        # fallback uniform if missing or malformed
        probs = [1.0/num_choices]*num_choices
    # optional time
    m_time = re.search(r"Time:\s*([0-9]+)\s*s", text, re.I)
    t_used = float(m_time.group(1)) if m_time else float(CFG.default_timelimit_sec)
    # normalize probs
    s = sum(probs)
    probs = [p/s if s>0 else 1.0/num_choices for p in probs]
    return letter, probs, t_used

def brier_score(probs: List[float], correct_idx: int) -> float:
    target = [0.0]*len(probs)
    target[correct_idx] = 1.0
    return sum((p - y)**2 for p,y in zip(probs, target)) / len(probs)

def rubric_check(item: Dict[str,Any], model_text: str) -> Dict[str,bool]:
    """
    Tiny heuristic rubric:
      - units_ok: looks for units tokens when numbers appear (CP/PS)
      - scope_ok: for CARS, requires a 'quote' or 'line' mention
      - graph_ok: if 'figure' or 'table' mentioned when prompt has them
    """
    sec = item["section"]
    passage = item["passage"]
    q = item["question"]
    # flags
    has_number = bool(re.search(r"\d", model_text))
    mentions_units = bool(re.search(r"(mol|M|s|J|kg|mL|L|Pa|N|V|A|Hz|°C|K)\b", model_text))
    mentions_quote = bool(re.search(r"(line\s*\d+|\".+\"|‘.+’|’.+’|\(.+?\))", model_text, re.I))
    prompt_has_figure = bool(re.search(r"(Figure|Table)", passage+q, re.I))
    mentions_figure = bool(re.search(r"(Figure|Table|Fig\.)", model_text, re.I))

    units_ok = (sec in ("CP","PS")) and (not has_number or mentions_units) or (sec in ("BB","CARS"))
    scope_ok = (sec != "CARS") or mentions_quote
    graph_ok = (not prompt_has_figure) or mentions_figure
    return {"units_ok": bool(units_ok), "scope_ok": bool(scope_ok), "graph_ok": bool(graph_ok)}

def choose_critic_label(item: Dict[str,Any], model_text: str, rubric: Dict[str,bool], is_correct: bool) -> str:
    if is_correct and all(rubric.values()):
        return "ok"
    # prioritize errors
    if item["section"] in ("CP","PS"):
        if not rubric["units_ok"]: return "math_units_issue"
        if not rubric["graph_ok"]: return "graph_misread"
        return "experimental_design"
    if item["section"] == "CARS":
        if not rubric["scope_ok"]: return "scope_overreach"
        return "no_passage_evidence"
    if item["section"] == "BB":
        return "biochem_pathway_confusion"
    return "no_passage_evidence"

# ================= METRICS / COMPARATORS =================
def metric_m(outputs: List[str], inputs: List[str]) -> torch.Tensor:
    """
    Primary task metric m \in [0,1]:
      correctness ± calibration bonus, minus timing penalty.
    """
    scores = []
    for out, inp in zip(outputs, inputs):
        item = json.loads(inp)
        choices = item["choices"]
        correct_idx = resolve_correct_index(item["answer"], choices)
        letter, probs, t_used = parse_model_output(out, len(choices))
        pred_idx = CHOICE_LETTERS.index(letter) if letter in CHOICE_LETTERS[:len(choices)] else -1
        correct = 1.0 if pred_idx == correct_idx else 0.0

        # timing penalty relative to timelimit
        T = float(item.get("timelimit_sec", CFG.default_timelimit_sec))
        over = max(0.0, (t_used - T)/max(T,1.0))  # 0 when on-time, 1 when double time
        time_pen = CFG.time_penalty_weight * min(1.0, over)

        # calibration term
        bs = brier_score(probs, correct_idx)
        calib = CFG.calibration_weight * (1.0 - bs)  # in [something around 0..0.5]

        base = correct - time_pen
        if correct > 0:
            base += calib
        else:
            base -= calib

        scores.append(float(max(0.0, min(1.0, base))))
    return torch.tensor(scores, device=CFG.device)

def best_known_score(inputs: List[str]) -> torch.Tensor:
    """
    Regret baseline: approximates 'ideal' rubric-satisfying answer.
    Use 0.9 target for now; you can plug a teacher model later.
    """
    return torch.tensor([0.9 for _ in inputs], device=CFG.device)

# simple rolling per-section peer medians (stubbed as constants; wire to checkpoints in production)
ROLLING_PEER = {("CP",1):0.55, ("CP",3):0.60, ("CARS",3):0.52, ("BB",3):0.58, ("PS",3):0.57}

def peer_scores(inputs: List[str], candidates_per_input: int = 3) -> torch.Tensor:
    meds = []
    for s in inputs:
        item = json.loads(s)
        key = (item["section"], item.get("difficulty",3))
        meds.append(ROLLING_PEER.get(key, 0.55))
    return torch.tensor(meds, device=CFG.device)

# ================= FAILURE REPLAY =================
class FailureReplay:
    def __init__(self, maxlen=CFG.max_failbuf):
        self.buf: List[Dict[str, Any]] = []
        self.maxlen = maxlen
    def add(self, record):
        self.buf.append(record)
        if len(self.buf) > self.maxlen:
            self.buf.pop(0)
    def sample_prompts(self, n):
        if not self.buf: return []
        return [random.choice(self.buf)["prompt_json"] for _ in range(n)]
    def repeated_error(self, rsn: str, prev_rsn: str) -> bool:
        return rsn == prev_rsn

FAILBUF = FailureReplay()

# ================= PPO MEMORY =================
@dataclass
class Traj:
    input_ids: torch.Tensor
    logp: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor

# ================= UTIL: logprobs & values =================
def logprobs_from_logits(logits, ids):
    logp = F.log_softmax(logits, dim=-1)
    lp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    mask = (ids != TOK.pad_id).float()
    seq_logp = (lp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return seq_logp

class TinyValueHead(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.lin = nn.Linear(d, 1)
    def forward(self, feats):
        v = self.lin(feats.mean(dim=1))
        return v.squeeze(-1)

# ================= TRAINER =================
class DiCTTrainer:
    def __init__(self):
        self.pi = PolicyModel().to(CFG.device)
        self.ref = PolicyModel().to(CFG.device)
        self.ref.load_state_dict(self.pi.state_dict())
        for p in self.ref.parameters(): p.requires_grad_(False)

        self.critic = CriticHead().to(CFG.device)
        self.vhead = TinyValueHead().to(CFG.device)
        self.opt = torch.optim.AdamW(
            list(self.pi.parameters()) + list(self.critic.parameters()) + list(self.vhead.parameters()),
            lr=CFG.lr, betas=CFG.betas, weight_decay=CFG.wd
        )
        # simple moving log for dashboard
        self.running = {"acc": [], "brier": [], "time_over": [], "reasons": {}}

    def features(self, input_ids, output_ids):
        with torch.no_grad():
            x = self.pi.tok_emb(input_ids)
            y = self.pi.tok_emb(output_ids)
            feats = torch.cat([x, y], dim=1)
        return feats

    def critique_then_rewrite(self, items: List[Dict[str,Any]], outputs: List[str]) -> Tuple[List[str], List[str], torch.Tensor]:
        # approval from critic head
        batch_ids = collate_text([format_mcat_prompt(it) + "\n\n" + o for it,o in zip(items, outputs)])
        with torch.no_grad():
            feats = self.pi.tok_emb(batch_ids)
            approval = self.critic(feats).squeeze(-1)  # [B], ~ approval prob

        rew_prompts, reasons = [], []
        for item, out, appr in zip(items, outputs, approval):
            letter, probs, t_used = parse_model_output(out, len(item["choices"]))
            correct_idx = resolve_correct_index(item["answer"], item["choices"]) 
            correct = (letter is not None and CHOICE_LETTERS.index(letter) == correct_idx)
            rb = rubric_check(item, out)
            reason = choose_critic_label(item, out, rb, correct)
            reasons.append(reason)
            if appr.item() < 0.6 or reason != "ok":
                # targeted overlay
                overlay = ""
                if reason == "math_units_issue":
                    overlay = "CRITIC FLAG: ignored units. Recompute showing units at each step; state the final units."
                elif reason == "graph_misread":
                    overlay = "CRITIC FLAG: graph misread. Cite the exact figure/table values used; point to the axis labels."
                elif reason == "scope_overreach":
                    overlay = "CRITIC FLAG: inference beyond scope. Use only claims explicitly supported by the quoted passage line."
                elif reason == "no_passage_evidence":
                    overlay = "CRITIC FLAG: no passage evidence. Quote the specific line supporting your claim."
                elif reason == "experimental_design":
                    overlay = "CRITIC FLAG: experimental design. Identify IV, DV, controls; then answer."
                elif reason == "biochem_pathway_confusion":
                    overlay = "CRITIC FLAG: biochem confusion. Name the exact pathway/step from the passage and use it in reasoning."
                else:
                    overlay = "CRITIC FLAG: quality. Tighten reasoning; follow format precisely."
                fix_prompt = format_mcat_prompt(item) + "\n\n" + overlay + "\nKeep the same output format."
                rew_prompts.append(fix_prompt)
            else:
                rew_prompts.append(format_mcat_prompt(item))
        return rew_prompts, reasons, approval

    def compute_rewards(self, items: List[Dict[str,Any]], outputs: List[str], critic_appr: torch.Tensor, reasons: List[str]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        # main metric
        inputs_as_json = [json.dumps(it) for it in items]
        m_scores = metric_m(outputs, inputs_as_json)
        best_scores = best_known_score(inputs_as_json)
        peer_median = peer_scores(inputs_as_json)

        comp_gap = (m_scores - peer_median)
        L_comp = torch.clamp(CFG.peer_margin_M - comp_gap, min=0.0)
        L_regret = torch.clamp(best_scores - m_scores, min=0.0)
        L_dis = 1.0 - critic_appr

        repeat_pen = torch.zeros_like(m_scores)
        for i, (it, rsn) in enumerate(zip(items, reasons)):
            sampled = FAILBUF.sample_prompts(1)
            if sampled and rsn != "ok":  # simplistic repeat detector
                repeat_pen[i] = 1.0

        neg = CFG.lambda_comp * L_comp + CFG.lambda_regret * L_regret + CFG.lambda_dis * L_dis + CFG.lambda_repeat * repeat_pen
        reward = (m_scores - neg).detach()

        # for dashboard
        dash = self._dash_stats(items, outputs)
        return reward, repeat_pen.detach(), dash

    def _dash_stats(self, items: List[Dict[str,Any]], outputs: List[str]) -> Dict[str,float]:
        accs, briers, overages = [], [], []
        for it, out in zip(items, outputs):
            choices = it["choices"]
            correct_idx = resolve_correct_index(it["answer"], choices)
            letter, probs, t_used = parse_model_output(out, len(choices))
            pred_idx = CHOICE_LETTERS.index(letter) if letter in CHOICE_LETTERS[:len(choices)] else -1
            accs.append(1.0 if pred_idx == correct_idx else 0.0)
            briers.append(brier_score(probs, correct_idx))
            T = float(it.get("timelimit_sec", CFG.default_timelimit_sec))
            over = max(0.0, (t_used - T)/max(T,1.0))
            overages.append(over)
        return {
            "acc": float(sum(accs)/max(1,len(accs))),
            "brier": float(sum(briers)/max(1,len(briers))),
            "time_over": float(sum(overages)/max(1,len(overages))),
        }

    def ppo_update(self, trajs: List[Traj]):
        input_ids = torch.cat([t.input_ids for t in trajs], dim=0)
        old_logp   = torch.cat([t.logp for t in trajs], dim=0)
        returns    = torch.cat([t.returns for t in trajs], dim=0)
        adv        = torch.cat([t.advantage for t in trajs], dim=0)

        logits = self.pi(input_ids)
        logp = logprobs_from_logits(logits, input_ids)
        ratio = torch.exp(logp - old_logp)

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1-CFG.cliprange, 1+CFG.cliprange) * adv
        ppo_loss = -torch.min(unclipped, clipped).mean()

        with torch.no_grad():
            feats = self.pi.tok_emb(input_ids)
        values = self.vhead(feats)
        vloss = F.mse_loss(values, returns)

        with torch.no_grad():
            ref_logits = self.ref(input_ids)
            ref_logp = logprobs_from_logits(ref_logits, input_ids)
        kl = (ref_logp - logp).mean()
        kl_pen = CFG.kl_coef * kl

        loss = ppo_loss + 0.5*vloss + kl_pen

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        self.opt.step()

        return {"loss": float(loss.item()), "ppo": float(ppo_loss.item()), "vloss": float(vloss.item()), "kl": float(kl.item())}

    def rollout_batch(self, items: List[Dict[str,Any]]) -> Tuple[List[Traj], Dict[str,float], List[str]]:
        prompts = [format_mcat_prompt(it) for it in items]
        input_ids = collate_text(prompts)
        with torch.no_grad():
            gen_ids = self.pi.generate(input_ids, max_len=min(CFG.max_len, input_ids.size(1)+192))
            logits = self.pi(input_ids)
            logp = logprobs_from_logits(logits, input_ids)
            feats = self.pi.tok_emb(gen_ids)
            values = self.vhead(feats)

        outputs_txt = [TOK.decode(ids.tolist()) for ids in gen_ids]
        # critique -> rewrite
        rew_prompts, reasons, approval = self.critique_then_rewrite(items, outputs_txt)
        edited_ids = collate_text(rew_prompts)
        with torch.no_grad():
            edited_gen = self.pi.generate(edited_ids, max_len=min(CFG.max_len, edited_ids.size(1)+192))
        edited_txt = [TOK.decode(ids.tolist()) for ids in edited_gen]

        # pick better (bandit)
        base_scores = metric_m(outputs_txt, [json.dumps(x) for x in items])
        edit_scores = metric_m(edited_txt, [json.dumps(x) for x in items])
        pick_edit = (edit_scores > base_scores).float()
        final_texts = [e if pe > 0 else o for o,e,pe in zip(outputs_txt, edited_txt, pick_edit)]

        rewards, repeat_pen, dash = self.compute_rewards(items, final_texts, approval, reasons)

        with torch.no_grad():
            adv = rewards - values
            returns = rewards

        # log failures
        for it, out, rsn, r in zip(items, final_texts, reasons, rewards.tolist()):
            if r < 0.35:
                FAILBUF.add({
                    "prompt_json": json.dumps(it),
                    "reason": rsn,
                    "output": out
                })

        traj = Traj(input_ids=input_ids, logp=logp, value=values.detach(), reward=rewards, advantage=adv, returns=returns)
        return [traj], dash, reasons

    def train(self, item_source):
        step = 0
        while step < CFG.max_steps:
            # sample items; mix in failure replay JSON strings
            clean_items = [item_source() for _ in range(CFG.bsz)]
            if random.random() < CFG.failure_replay_prob:
                replay = FAILBUF.sample_prompts(CFG.bsz // 2)
                if replay:
                    for i, rj in enumerate(replay):
                        clean_items[i] = json.loads(rj)

            trajs, dash, reasons = self.rollout_batch(clean_items)
            logs = self.ppo_update(trajs)
            step += 1

            # update dashboard running stats
            self.running["acc"].append(dash["acc"])
            self.running["brier"].append(dash["brier"])
            self.running["time_over"].append(dash["time_over"])
            for r in reasons:
                self.running["reasons"][r] = self.running["reasons"].get(r, 0) + 1

            if step % 50 == 0:
                top_reasons = sorted(self.running["reasons"].items(), key=lambda x: -x[1])[:3]
                tr_str = ", ".join(f"{k}:{v}" for k,v in top_reasons) if top_reasons else "—"
                print(f"[{step}] loss={logs['loss']:.4f} ppo={logs['ppo']:.4f} v={logs['vloss']:.4f} kl={logs['kl']:.4f} | "
                      f"acc7={sum(self.running['acc'][-50:])/max(1,len(self.running['acc'][-50:])):.3f} "
                      f"brier7={sum(self.running['brier'][-50:])/max(1,len(self.running['brier'][-50:])):.3f} "
                      f"over7={sum(self.running['time_over'][-50:])/max(1,len(self.running['time_over'][-50:])):.3f} "
                      f"| buf={len(FAILBUF.buf)} | top_err={tr_str}")

            # curriculum: raise penalties as model stabilizes
            if step % 1000 == 0:
                CFG.lambda_comp *= 1.05
                CFG.lambda_regret *= 1.05
                CFG.lambda_dis *= 1.05
                CFG.lambda_repeat *= 1.02
                CFG.peer_margin_M = min(0.12, CFG.peer_margin_M * 1.05)

# ================= DATALOADER STUB =================
_FAKE_PASSAGE = "Figure 2 shows a linear increase in current with voltage. The enzyme rate follows Michaelis-Menten kinetics at 37°C."
_FAKE_Q = "Based on Figure 2, doubling voltage would most likely produce what change in current?"
_FAKE_CHOICES = ["Decreases by half", "No change", "Doubles", "Quadruples"]

def sample_mcat_item() -> Dict[str,Any]:
    # Replace with a real iterator over your MCAT JSON dataset.
    sec = random.choice(SECTIONS)
    if sec == "CARS":
        passage = "The author argues that aesthetic value arises from communal practices rather than individual genius."
        q = "The author's view implies which of the following about solitary artists?"
        choices = ["They cannot create art.", "Their work lacks any value.", "Community context informs value.", "Genius guarantees acclaim."]
        ans = "C"
    else:
        passage = _FAKE_PASSAGE
        q = _FAKE_Q
        choices = _FAKE_CHOICES
        ans = "C"
    return {
        "section": sec,
        "passage": passage,
        "question": q,
        "choices": choices,
        "answer": ans,
        "rationale_key": "Use the figure trend; for CARS, cite lines supporting inference.",
        "skills": ["units","graph"] if sec in ("CP","PS") else ["scope"],
        "difficulty": 3,
        "timelimit_sec": CFG.default_timelimit_sec
    }

# ================= ENTRY =================
if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    trainer = DiCTTrainer()
    trainer.train(sample_mcat_item)
