# -*- coding: utf-8 -*-
"""Empirical study for Adaptive Language Pivoting.

1. Build real JA-EN and AR-EN parallel pairs (Tatoeba).
2. Measure token inflation ratios r = tokens(src)/tokens(en) for the
   GPT-4 (cl100k_base) and GPT-4o (o200k_base) tokenizers.
3. Train the per-language English-token predictor; evaluate on held-out data.
4. Simulate realistic requests (bundles of ~400 EN tokens in, 300 EN tokens
   out) and compare four policies: always-direct, always-pivot,
   adaptive (predictor-based router), oracle (true EN counts).
5. Emit results.json + trained predictor weights.

Run: python -X utf8 scripts/run_study.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import tiktoken

from langpivot.corpus import build_pairs
from langpivot.costmodel import PRICES, CostModel
from langpivot.predictor import EnTokenPredictor
from langpivot.router import PivotRouter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

LANGS = {
    "ja": ("jpn_sentences.tsv.bz2", "jpn-eng_links.tsv.bz2"),
    "ar": ("ara_sentences.tsv.bz2", "ara-eng_links.tsv.bz2"),
}
ENCODINGS = ["cl100k_base", "o200k_base"]
MAIN_ENC = "o200k_base"  # tokenizer of the simulated main model (gpt-4o)

REQ_EN_IN = 400    # simulated request size, English input tokens
REQ_EN_OUT = 300   # simulated answer size, English output tokens


def measure(pairs, enc):
    src_tot = en_tot = 0
    ratios = []
    per_pair = []
    for s, e in pairs:
        ts = len(enc.encode(s))
        te = len(enc.encode(e))
        src_tot += ts
        en_tot += te
        per_pair.append((ts, te))
        if te:
            ratios.append(ts / te)
    ratios.sort()
    n = len(ratios)
    return {
        "pairs": n,
        "src_tokens": src_tot,
        "en_tokens": en_tot,
        "ratio_corpus": src_tot / en_tot,
        "ratio_median": ratios[n // 2],
        "ratio_p25": ratios[n // 4],
        "ratio_p75": ratios[3 * n // 4],
    }, per_pair


def bundle_requests(pairs, enc, target_en_tokens):
    """Concatenate consecutive sentence pairs into realistic request-sized
    chunks: returns list of (src_text, en_text, src_tokens, en_tokens)."""
    out = []
    buf_s, buf_e = [], []
    en_count = 0
    for s, e in pairs:
        buf_s.append(s)
        buf_e.append(e)
        en_count += len(enc.encode(e))
        if en_count >= target_en_tokens:
            src_text = "\n".join(buf_s)
            en_text = "\n".join(buf_e)
            out.append(
                (src_text, en_text, len(enc.encode(src_text)), len(enc.encode(en_text)))
            )
            buf_s, buf_e = [], []
            en_count = 0
    return out


def simulate(lang, requests, predictor, r_out, main="gpt-4o", translator="gpt-4o-mini"):
    cm = CostModel(PRICES[main], PRICES[translator])
    router = PivotRouter(
        main_model=main, translator_model=translator,
        predictor=predictor, r_out={lang: r_out},
    )
    totals = {"direct": 0.0, "pivot": 0.0, "adaptive": 0.0, "oracle": 0.0}
    pivoted = agree = 0
    for src_text, _en_text, src_tok, en_tok in requests:
        r_in_true = src_tok / en_tok
        costs_true = cm.pivot_cost(en_tok, REQ_EN_OUT, r_in_true, r_out)
        totals["direct"] += costs_true.direct
        totals["pivot"] += costs_true.pivot
        oracle_pivot = costs_true.pivot < costs_true.direct
        totals["oracle"] += min(costs_true.direct, costs_true.pivot)
        decision = router.decide(src_text, expected_en_tokens_out=REQ_EN_OUT)
        # realized cost of the adaptive policy uses TRUE token counts
        if decision.route == "pivot":
            pivoted += 1
            totals["adaptive"] += costs_true.pivot
        else:
            totals["adaptive"] += costs_true.direct
        if (decision.route == "pivot") == oracle_pivot:
            agree += 1
    n = len(requests)
    return {
        "requests": n,
        "main_model": main,
        "translator_model": translator,
        "en_tokens_out_assumed": REQ_EN_OUT,
        "cost_always_direct_usd": totals["direct"],
        "cost_always_pivot_usd": totals["pivot"],
        "cost_adaptive_usd": totals["adaptive"],
        "cost_oracle_usd": totals["oracle"],
        "savings_adaptive_pct": 100 * (1 - totals["adaptive"] / totals["direct"]),
        "savings_always_pivot_pct": 100 * (1 - totals["pivot"] / totals["direct"]),
        "savings_oracle_pct": 100 * (1 - totals["oracle"] / totals["direct"]),
        "pivoted_fraction": pivoted / n,
        "oracle_agreement": agree / n,
    }


def main():
    encs = {name: tiktoken.get_encoding(name) for name in ENCODINGS}
    results = {"setup": {
        "corpus": "Tatoeba parallel pairs (CC-BY 2.0 FR)",
        "request_en_tokens_in": REQ_EN_IN,
        "request_en_tokens_out": REQ_EN_OUT,
        "prices_usd_per_mtok": {
            k: [v.input_per_m, v.output_per_m] for k, v in PRICES.items()
        },
    }, "languages": {}}

    predictor = EnTokenPredictor()

    for lang, (src_file, links_file) in LANGS.items():
        print(f"[{lang}] building parallel pairs ...", flush=True)
        pairs = build_pairs(
            os.path.join(DATA, src_file),
            os.path.join(DATA, "eng_sentences.tsv.bz2"),
            os.path.join(DATA, links_file),
            max_pairs=60000,
        )
        print(f"[{lang}] {len(pairs)} pairs", flush=True)
        entry = {"pairs": len(pairs), "tokenizers": {}}

        for enc_name, enc in encs.items():
            stats, _ = measure(pairs, enc)
            entry["tokenizers"][enc_name] = stats
            print(f"[{lang}] {enc_name}: corpus ratio r = {stats['ratio_corpus']:.3f} "
                  f"(median {stats['ratio_median']:.3f})", flush=True)

        # train/test split for the predictor (target: MAIN_ENC English tokens)
        enc = encs[MAIN_ENC]
        split = int(len(pairs) * 0.8)
        train, test = pairs[:split], pairs[split:]
        fit = predictor.fit(
            lang, [s for s, _ in train], [len(enc.encode(e)) for _, e in train]
        )
        # held-out evaluation
        preds = [predictor.predict(lang, s) for s, _ in test]
        trues = [len(enc.encode(e)) for _, e in test]
        mape = 100 * sum(abs(p - t) / t for p, t in zip(preds, trues) if t) / len(trues)
        entry["predictor"] = {"train": fit, "test_mape_pct": mape, "test_n": len(test)}
        print(f"[{lang}] predictor: train R2={fit['r2']:.3f}, held-out MAPE={mape:.1f}%",
              flush=True)

        # request-level simulation on held-out pairs
        requests = bundle_requests(test, enc, REQ_EN_IN)
        r_out = entry["tokenizers"][MAIN_ENC]["ratio_corpus"]
        entry["simulation"] = {}
        for main_m, trans_m in [("gpt-4o", "gpt-4o-mini"),
                                ("gpt-4-turbo", "gpt-4o-mini"),
                                ("gpt-4o", "local-mt")]:
            sim = simulate(lang, requests, predictor, r_out, main_m, trans_m)
            entry["simulation"][f"{main_m}+{trans_m}"] = sim
            print(f"[{lang}] {main_m}+{trans_m}: adaptive saves "
                  f"{sim['savings_adaptive_pct']:.1f}% "
                  f"(always-pivot {sim['savings_always_pivot_pct']:.1f}%, "
                  f"oracle {sim['savings_oracle_pct']:.1f}%), "
                  f"pivoted {100*sim['pivoted_fraction']:.0f}%", flush=True)

        results["languages"][lang] = entry

    # break-even table (price-only; language-independent)
    results["breakeven_r"] = {}
    for main_m in ("gpt-4o", "gpt-4-turbo", "claude-opus-4-8", "claude-sonnet-5"):
        for trans_m in ("gpt-4o-mini", "claude-haiku-4-5", "local-mt"):
            cm = CostModel(PRICES[main_m], PRICES[trans_m])
            results["breakeven_r"][f"{main_m}+{trans_m}"] = {
                "out/in=0.25": cm.breakeven_ratio(0.25),
                "out/in=0.75": cm.breakeven_ratio(0.75),
                "out/in=2.0": cm.breakeven_ratio(2.0),
            }

    predictor.save(os.path.join(ROOT, "predictor_weights.json"))
    out = os.path.join(ROOT, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] results -> {out}")


if __name__ == "__main__":
    main()
