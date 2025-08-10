#!/usr/bin/env python3
import argparse, json, re, sys
from typing import Any, List

SECTIONS = {"CP","CARS","BB","PS"}
CHOICE_LETTERS = ["A","B","C","D","E"]


def resolve_correct_index(answer: Any, choices: List[Any]) -> int:
    n = len(choices)
    try:
        if isinstance(answer, str):
            s = answer.strip()
            if len(s) == 1 and s.upper() in CHOICE_LETTERS[:n]:
                return CHOICE_LETTERS.index(s.upper())
            if re.fullmatch(r"\d+", s):
                i = int(s)
                if 0 <= i < n:
                    return i
                if 1 <= i <= n:
                    return i - 1
            try:
                return choices.index(answer)
            except ValueError:
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
    return -1


def validate_item(obj: dict, lineno: int) -> List[str]:
    errs = []
    # required fields
    for k in ("section","passage","question","choices","answer"):
        if k not in obj:
            errs.append(f"line {lineno}: missing field '{k}'")
            return errs
    # section
    if obj["section"] not in SECTIONS:
        errs.append(f"line {lineno}: invalid section '{obj['section']}', must be one of {sorted(SECTIONS)}")
    # passage/question
    if not isinstance(obj["passage"], str) or not obj["passage"].strip():
        errs.append(f"line {lineno}: 'passage' must be a non-empty string")
    if not isinstance(obj["question"], str) or not obj["question"].strip():
        errs.append(f"line {lineno}: 'question' must be a non-empty string")
    # choices
    ch = obj["choices"]
    if not isinstance(ch, list) or not (2 <= len(ch) <= 5):
        errs.append(f"line {lineno}: 'choices' must be a list with 2..5 options")
    else:
        for i, c in enumerate(ch):
            if not isinstance(c, str) or not c.strip():
                errs.append(f"line {lineno}: choices[{i}] must be a non-empty string")
                break
    # answer
    if isinstance(ch, list) and 2 <= len(ch) <= 5:
        idx = resolve_correct_index(obj["answer"], ch)
        if idx < 0 or idx >= len(ch):
            errs.append(f"line {lineno}: 'answer' not resolvable to an index in 0..{len(ch)-1}; got {obj['answer']}")
    # difficulty (optional)
    if "difficulty" in obj:
        try:
            d = int(obj["difficulty"])
            if d < 1 or d > 5:
                errs.append(f"line {lineno}: 'difficulty' should be in 1..5 (got {obj['difficulty']})")
        except Exception:
            errs.append(f"line {lineno}: 'difficulty' must be an integer (got {obj['difficulty']})")
    # timelimit_sec (optional)
    if "timelimit_sec" in obj:
        try:
            t = int(obj["timelimit_sec"])
            if t < 10 or t > 600:
                errs.append(f"line {lineno}: 'timelimit_sec' looks unusual ({t}); expected 10..600")
        except Exception:
            errs.append(f"line {lineno}: 'timelimit_sec' must be an integer (got {obj['timelimit_sec']})")
    return errs


def main():
    ap = argparse.ArgumentParser(description="Validate MCAT JSONL dataset for DiCT-PPO trainer")
    ap.add_argument("--path", required=True, help="Path to JSONL file")
    args = ap.parse_args()

    errors = 0
    total = 0
    try:
        with open(args.path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"line {lineno}: JSON parse error: {e}", file=sys.stderr)
                    errors += 1
                    continue
                total += 1
                errs = validate_item(obj, lineno)
                for e in errs:
                    print(e, file=sys.stderr)
                errors += len(errs)
    except FileNotFoundError:
        print(f"file not found: {args.path}", file=sys.stderr)
        return 2

    if errors == 0:
        print(f"OK: {total} items validated, 0 errors")
        return 0
    else:
        print(f"ERROR: {errors} issues found across {total} items", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
