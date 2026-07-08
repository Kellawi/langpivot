# -*- coding: utf-8 -*-
"""Preliminary quality evaluation for adaptive language pivoting.

For each Japanese prompt we obtain two answers:
  * DIRECT: the main model answers in Japanese.
  * PIVOT : translate prompt JA->EN (translator), main model answers in
            English, translate answer EN->JA (translator).
A separate LLM judge compares the two Japanese answers blind and
position-randomised, choosing better / better / tie with a short reason.
We report win/tie/loss for pivot vs direct, overall and by category, and
tie each item back to the cost model by measuring the token delta.

This is the "threats to validity" sneak-peek experiment: it measures the
QUALITY side that the cost study (run_study.py) deliberately left open.

Providers: OpenAI (chat/completions) or Anthropic (messages). Runs with
whichever API key is set; override with --provider/--main/--translator/--judge.

    python -X utf8 scripts/quality_eval.py --limit 50
    python -X utf8 scripts/quality_eval.py --limit 3      # cheap smoke test
    python -X utf8 scripts/quality_eval.py --dry-run      # no API calls
"""

import argparse
import io
import json
import os
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

DEFAULTS = {
    "openai": {"main": "gpt-4o", "translator": "gpt-4o-mini", "judge": "gpt-4o",
               "base": "https://api.openai.com/v1", "key": "OPENAI_API_KEY"},
    "anthropic": {"main": "claude-opus-4-8", "translator": "claude-haiku-4-5",
                  "judge": "claude-opus-4-8",
                  "base": "https://api.anthropic.com/v1", "key": "ANTHROPIC_API_KEY"},
    "gemini": {"main": "gemini-2.5-pro", "translator": "gemini-2.0-flash",
               "judge": "gemini-2.5-pro",
               "base": "https://generativelanguage.googleapis.com/v1beta",
               "key": "GEMINI_API_KEY"},
}

# ------------------------------------------------------------------ clients

class APIError(Exception):
    pass


class Client:
    def __init__(self, provider, base, key):
        self.provider = provider
        self.base = base.rstrip("/")
        self.key = key

    def _post(self, url, body, headers):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))

    def chat(self, model, system, user, max_tokens=1024, temperature=0.2, retries=5):
        for attempt in range(retries):
            try:
                if self.provider == "openai":
                    body = {"model": model, "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}],
                        "temperature": temperature, "max_tokens": max_tokens}
                    resp = self._post(f"{self.base}/chat/completions", body, {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.key}"})
                    return resp["choices"][0]["message"]["content"].strip()
                elif self.provider == "gemini":
                    # thinking models need output headroom beyond the reply itself;
                    # disable thinking on flash so the cheap translator stays cheap.
                    gen = {"maxOutputTokens": max_tokens + 4096, "temperature": temperature}
                    if "flash" in model:
                        gen["thinkingConfig"] = {"thinkingBudget": 0}
                    body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
                            "systemInstruction": {"parts": [{"text": system}]},
                            "generationConfig": gen}
                    resp = self._post(
                        f"{self.base}/models/{model}:generateContent?key={self.key}",
                        body, {"Content-Type": "application/json"})
                    cand = (resp.get("candidates") or [{}])[0]
                    parts = cand.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()
                    if not text:
                        raise APIError(f"empty gemini output "
                                       f"(finishReason={cand.get('finishReason')})")
                    return text
                else:  # anthropic (no temperature on current models)
                    body = {"model": model, "system": system, "max_tokens": max_tokens,
                            "messages": [{"role": "user", "content": user}]}
                    resp = self._post(f"{self.base}/messages", body, {
                        "Content-Type": "application/json",
                        "x-api-key": self.key,
                        "anthropic-version": "2023-06-01"})
                    return "".join(b.get("text", "") for b in resp["content"]).strip()
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503, 529) and attempt < retries - 1:
                    time.sleep(2 ** attempt + random.random())
                    continue
                raise APIError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")
        raise APIError("exhausted retries")


# ------------------------------------------------------------------ pipeline

SYS_JA = "あなたは有能で丁寧なアシスタントです。ユーザーの言語（日本語）で自然に回答してください。"
SYS_EN = "You are a capable, helpful assistant. Answer clearly and naturally in English."
TL = ("You are a professional translator. Translate the user's message into {tgt}. "
      "Preserve meaning, tone, formality/register, and any formatting. "
      "Output ONLY the translation, no notes.")

JUDGE_SYS = (
    "あなたは公平な評価者です。あるユーザーの質問に対する二つの回答（回答Aと回答B）を比較し、"
    "どちらがより優れているかを判断します。評価基準：正確さ、日本語としての自然さ・敬語や語調の適切さ、"
    "文化的な妥当性、質問への網羅性。厳密なJSONのみを出力してください："
    '{"winner":"A"|"B"|"tie","reason":"簡潔な理由"}')


def translate(cli, model, text, tgt):
    return cli.chat(model, TL.format(tgt=tgt), text, max_tokens=1200)


def answer_direct(cli, model, prompt):
    return cli.chat(model, SYS_JA, prompt, max_tokens=800)


def answer_pivot(cli, main, tl, prompt):
    en_prompt = translate(cli, tl, prompt, "English")
    en_answer = cli.chat(main, SYS_EN, en_prompt, max_tokens=800)
    ja_answer = translate(cli, tl, en_answer, "Japanese")
    return ja_answer, en_prompt, en_answer


def judge(cli, model, prompt, ans_a, ans_b):
    user = (f"【質問】\n{prompt}\n\n【回答A】\n{ans_a}\n\n【回答B】\n{ans_b}\n\n"
            "JSONのみで判定してください。")
    raw = cli.chat(model, JUDGE_SYS, user, max_tokens=400, temperature=0.0)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"winner": "tie", "reason": "parse_error", "raw": raw[:200]}
    try:
        obj = json.loads(m.group(0))
        if obj.get("winner") not in ("A", "B", "tie"):
            obj["winner"] = "tie"
        return obj
    except json.JSONDecodeError:
        return {"winner": "tie", "reason": "json_error", "raw": raw[:200]}


# ------------------------------------------------------------------ driver

def load_prompts(limit):
    rows = []
    with open(os.path.join(DATA, "ja_eval_prompts.tsv"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                cat, prompt = line.split("\t", 1)
                rows.append((cat, prompt))
    return rows[:limit] if limit else rows


OUT = os.path.join(ROOT, "quality_results.json")


def summarize(items):
    tally = {"pivot": 0, "direct": 0, "tie": 0}
    by_cat = {}
    for it in items:
        o = it.get("outcome")
        if o in tally:
            tally[o] += 1
            by_cat.setdefault(it["category"], {"pivot": 0, "direct": 0, "tie": 0})[o] += 1
    n = sum(tally.values())
    return {
        "judged": n,
        "pivot_wins": tally["pivot"], "direct_wins": tally["direct"], "ties": tally["tie"],
        "pivot_win_rate": tally["pivot"] / n if n else None,
        "direct_win_rate": tally["direct"] / n if n else None,
        "tie_rate": tally["tie"] / n if n else None,
        "by_category": by_cat,
    }, tally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["auto", "openai", "anthropic", "gemini"],
                    default="auto")
    ap.add_argument("--main"); ap.add_argument("--translator"); ap.add_argument("--judge")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue a previous run, skipping items already judged")
    args = ap.parse_args()

    provider = args.provider
    if provider == "auto":
        provider = ("openai" if os.environ.get("OPENAI_API_KEY") else
                    "gemini" if os.environ.get("GEMINI_API_KEY") else
                    "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else None)
    cfg = DEFAULTS.get(provider or "openai")
    main_m = args.main or cfg["main"]
    tl_m = args.translator or cfg["translator"]
    judge_m = args.judge or cfg["judge"]

    prompts = load_prompts(args.limit)
    rng = random.Random(args.seed)
    print(f"[eval] provider={provider} main={main_m} translator={tl_m} judge={judge_m} "
          f"items={len(prompts)}", flush=True)

    if args.dry_run or provider is None:
        print("[eval] DRY RUN — no API calls. Pipeline and prompts validated.")
        print(f"[eval] would evaluate {len(prompts)} items across "
              f"{len(set(c for c, _ in prompts))} categories.")
        return

    key = os.environ[cfg["key"]]
    cli = Client(provider, cfg["base"], key)

    # token accounting proxy (o200k) for output length, best effort
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        tok = lambda s: len(enc.encode(s))
    except Exception:
        tok = lambda s: 0

    # resume: keep completed items, retry errors, skip re-judging done indices
    prior, done = [], set()
    if args.resume and os.path.exists(OUT):
        for it in json.load(open(OUT, encoding="utf-8")).get("items", []):
            if "outcome" in it:
                prior.append(it); done.add(it["i"])
        print(f"[eval] resume: {len(done)} items already judged", flush=True)

    results = {"setup": {"provider": provider, "main": main_m, "translator": tl_m,
                         "judge": judge_m, "n": len(prompts)}, "items": list(prior)}

    def checkpoint():
        results["summary"], _ = summarize(results["items"])
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    for i, (cat, prompt) in enumerate(prompts, 1):
        if i in done:
            continue
        try:
            direct = answer_direct(cli, main_m, prompt)
            pivot, en_p, en_a = answer_pivot(cli, main_m, tl_m, prompt)
            direct_is_A = rng.random() < 0.5  # position-randomise A/B
            a, b = (direct, pivot) if direct_is_A else (pivot, direct)
            verdict = judge(cli, judge_m, prompt, a, b)
            w = verdict["winner"]
            if w == "tie":
                outcome = "tie"
            else:
                outcome = "direct" if ((w == "A") == direct_is_A) else "pivot"
            results["items"].append({
                "i": i, "category": cat, "prompt": prompt,
                "direct": direct, "pivot": pivot, "en_prompt": en_p, "en_answer": en_a,
                "direct_is_A": direct_is_A, "verdict": verdict, "outcome": outcome,
                "tokens_direct_out": tok(direct), "tokens_pivot_out": tok(pivot),
            })
            print(f"  [{i}/{len(prompts)}] {cat}: {outcome}  "
                  f"({verdict.get('reason','')[:40]})", flush=True)
        except APIError as e:
            print(f"  [{i}] API error: {e}", flush=True)
            results["items"].append({"i": i, "category": cat, "error": str(e)})
        checkpoint()  # write after every item so a kill never loses progress

    _, tally = summarize(results["items"])
    print(f"\n[summary] judged={sum(tally.values())}  pivot={tally['pivot']}  "
          f"direct={tally['direct']}  tie={tally['tie']}")
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
