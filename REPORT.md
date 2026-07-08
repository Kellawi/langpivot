# Adaptive Language Pivoting for Reducing LLM API Cost — Measured Results

**TL;DR.** Pivoting Japanese/Arabic requests through English before calling an
expensive LLM API saves real money — **26% for Japanese and 12% for Arabic on
GPT-4o with gpt-4o-mini as the translator, 33–37% with a free local MT model**
— but *only* because the translator is ~16× cheaper than the main model. The
decision rule is exactly a break-even inequality between the language's token
inflation ratio `r` and the main/translator price gap; with a translator only
2–3× cheaper, pivoting **loses money** for these languages on modern
tokenizers. Cost is now measured; **answer quality under pivoting is the open
question the paper must still evaluate.**

All numbers below are measured (2026-07-04) by [scripts/run_study.py](scripts/run_study.py)
on real Tatoeba parallel corpora (60,000 JA–EN and 46,518 AR–EN sentence
pairs) with the actual GPT tokenizers via tiktoken. Raw output: `results.json`.

## 1. The problem, measured: token inflation ratios

`r` = tokens(source language) / tokens(English translation of the same content):

| Language | cl100k_base (GPT-4/3.5 era) | o200k_base (GPT-4o) |
|---|---|---|
| Japanese | **2.08** (median 2.10) | **1.57** (median 1.57) |
| Arabic | **2.47** (median 2.40) | **1.30** (median 1.27) |

Two observations. First, the "token tax" is real: the same content costs
1.3–2.5× more tokens in JA/AR than in English. Second, **OpenAI already
closed much of the gap** when moving from cl100k to o200k (JA 2.08→1.57,
AR 2.47→1.30) — so part of the historical pain is solved by just using
newer models, and any pivoting paper must use current-tokenizer numbers,
not the older ~2.5× folklore.

## 2. The economics: pivot cost vs direct cost

Pivot = translate-in (cheap model) + main call in English + translate-out
(cheap model). With input price `P_i`/output `P_o` for the main model,
`p_i`/`p_o` for the translator, and answer/prompt length ratio `k`, pivoting
is cheaper exactly when `r > r*` with

```
r* = (p_o + P_i + P_o·k + p_i·k) / ((P_i + P_o·k) − (p_i + p_o·k))
```

Measured break-even ratios `r*` (at k = 0.75):

| Main model | Translator | r* | JA (1.57) pivots? | AR (1.30) pivots? |
|---|---|---|---|---|
| gpt-4o ($2.5/$10) | gpt-4o-mini ($0.15/$0.60) | **1.14** | ✅ yes | ✅ yes |
| gpt-4o | free local MT | **1.00** | ✅ yes | ✅ yes |
| gpt-4o | claude-haiku-4-5 ($1/$5) | **3.00** | ❌ no | ❌ no |
| claude-opus-4-8 ($5/$25) | gpt-4o-mini | **1.06** | ✅* | ✅* |
| claude-sonnet-5 ($3/$15) | claude-haiku-4-5 | **2.11** | ❌* | ❌* |
| any model | itself as translator | **∞ (never)** | ❌ | ❌ |

\* Claude's tokenizer is not public; the ✅/❌ assumes JA/AR inflation similar
to o200k. Before deploying against Claude, measure `r` with Anthropic's
`count_tokens` API — never with tiktoken.

**The core insight:** pivoting is not "translation saves tokens"; it is an
arbitrage between the token-inflation ratio and the price gap. If the
translator is less than ~2–3× cheaper than the main model, the arbitrage
disappears for every language measured here.

## 3. Policy simulation on real data

Held-out parallel pairs bundled into realistic requests (~400 English tokens
in, 300 out), four routing policies compared:

| Language | Main + translator | Always-direct | Always-pivot | **Adaptive router** | Oracle |
|---|---|---|---|---|---|
| Japanese | gpt-4o + mini | 0% (baseline) | 26.0% | **26.0%** | 26.0% |
| Japanese | gpt-4-turbo + mini | 0% | 33.3% | **33.3%** | 33.3% |
| Japanese | gpt-4o + local MT | 0% | 36.5% | **36.5%** | 36.5% |
| Arabic | gpt-4o + mini | 0% | 11.6% | **11.6%** | 11.6% |
| Arabic | gpt-4-turbo + mini | 0% | 19.6% | **19.6%** | 19.6% |
| Arabic | gpt-4o + local MT | 0% | 23.2% | **23.2%** | 23.2% |

(Percentages = cost saved vs always-direct. Router = trained EN-token
predictor + break-even test; Oracle = decisions using true EN token counts.)

**Honest finding about the "query optimizer":** for *monolingual* JA/AR
traffic on a *fixed* (main, translator) pair, the adaptive router matches the
oracle but so does blind always-pivot — because at r = 1.3–1.6 versus
r* = 1.14, every request clears the break-even bar. The per-request decision
is nearly static. Where the adaptive layer actually earns its name:

1. **Mixed-language traffic** — the skip-if-English rule (your original
   design) avoids pointless translation calls; script detection is free.
2. **Deployment changes** — swap the translator to something only 2× cheaper
   (or the main model to a cheaper one) and the same router correctly flips
   to always-direct (see the gpt-4o + haiku row). Hard-coded "always
   translate" middleware silently starts burning money.
3. **Borderline pairs** — Arabic on o200k (1.30) sits close enough to some
   break-evens that per-request variance matters.
4. **Cost forecasting** — the predictor (held-out MAPE 17–20%, R² 0.70–0.82)
   prices each request before committing, which is what makes the system
   auditable.

So the paper's contribution is best framed as the **decision framework +
measured break-even surface**, with the learned predictor as the component
that makes deciding-without-translating possible — not as a claim that
per-request ML routing beats a constant policy on monolingual traffic.

## 4. The quality side, now measured (preliminary)

Everything in §1–3 is cost. The pivot path adds two machine translations, with
the classic risks: register/formality (敬語), culturally bound references,
intended ambiguity, wordplay, and errors compounding across two hops. We ran a
**preliminary LLM-as-judge study** (`scripts/quality_eval.py`, 2026-07-05): 50
Japanese prompts across 10 categories, each answered **direct** (gpt-4o in
Japanese) vs **pivot** (gpt-4o-mini translate → gpt-4o in English → translate
back), judged blind and position-randomized by gpt-4o in Japanese.

**Result: pivoting has a real quality cost — direct is preferred 52% (26/50) vs
28% (14/50) for pivot, 20% ties — and it lands exactly where predicted:**

| Band | Categories | Direct : Pivot : Tie |
|---|---|---|
| Pivot **hurts** | summarisation, cultural, business/keigo, factual | direct wins most |
| Mixed / neutral | how-to, advice, creative, nuance, explanation | roughly even |
| Pivot **safe** | reasoning | **5/5 ties** (language-invariant) |

So the translate-test intuition holds: reasoning survives pivoting (all ties),
while style- and culture-bound generation degrades. This confirms the design:
fold a **per-category quality prior into the router's margin** — pivot
reasoning/extraction/throughput traffic (the 26% JA saving is essentially
free), keep keigo/cultural/creative generation direct.

Caveats (stated in the paper too): the judge is the same model family as the
answerer (within-gpt-4o, not a human panel), per-category counts are small, and
only Japanese is covered. A larger human-rated, multi-language study is the
natural next step; back-translation adequacy scoring can add per-request
confidence routing. Full per-item data: `quality_results.json`.

## 5. Practical recommendations (deployable today)

1. **Write system prompts in English even for JA/AR products.** You author
   the system prompt — no translation call, no quality risk, and it's often
   the largest input block. This is free money before any pivoting.
2. **Combine with prompt caching, and mind the interaction.** A cached system
   prompt already gets ~90% off input; pivoting a cached block saves almost
   nothing extra. Pivot the *volatile* user content, cache the stable prefix.
3. **Half-pivot option**: prompt in English, instruct the model to answer in
   Japanese/Arabic directly. Saves the input-side inflation and the
   translate-out call (and its quality risk), keeps output cost native.
4. **Use the cheapest competent translator**: gpt-4o-mini-class (≥10× price
   gap) or a local MT model (NLLB/Marian — makes pivoting strictly
   cost-positive at any r > 1). Never translate with the main model.
5. **Accept the latency cost consciously**: pivot = 3 sequential calls,
   roughly 2–3× end-to-end latency. Wrong for interactive chat with short
   answers; right for batch/async and long-form generation.
6. **Re-measure when models change**: o200k halved the JA penalty vs cl100k.
   A future tokenizer could erase the arbitrage entirely — the router's
   pricing table and measured `r` are config, not constants.

## 6. Paper framing (for "Adaptive Language Pivoting for Reducing LLM API Cost")

The translate-test paradigm itself is old, and token-price inequality across
languages is documented (see "Do All Languages Cost the Same?" Ahia et al.
2023, and Petrov et al. 2023 on tokenizer unfairness — verify citations).
The defensible novelty here is: (i) the cost-aware routing formalization with
the closed-form break-even surface, (ii) deciding without translating via the
learned EN-token predictor, and (iii) the empirical result that the decision
is near-static per (language, model pair) — which reframes the contribution
from "ML router" to "measured decision framework with a skip rule". The
quality study in §4 is the missing experiment that would make it a full paper.

---
Bashar Kellawi — BasharKilawe@gmail.com
