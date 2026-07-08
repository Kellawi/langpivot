# langpivot — Adaptive Language Pivoting for Reducing LLM API Cost

Japanese and Arabic cost more per request on LLM APIs because their tokenizer
fertility is higher — the same content costs 1.3–2.5× the tokens of its
English equivalent (measured, see [REPORT.md](REPORT.md)). This package
implements a **cost-aware routing layer** ("query optimizer for tokens") that
decides per request whether to:

- **direct** — send the JA/AR text to the main model as-is, or
- **pivot** — translate to English with a cheap model, run the main model in
  English, translate the answer back.

The router skips everything that is already English, estimates the English
token count *without translating* (linear predictor trained on real parallel
corpora), and pivots only when the projected saving clears a break-even
margin. **Measured savings: 26% (JA) / 12% (AR) on gpt-4o with gpt-4o-mini as
translator; 33–37% with free local MT.** See [REPORT.md](REPORT.md) for the
full study, the break-even table, and the honest caveats (quality is not yet
evaluated; latency ~2–3×).

## Layout

| Path | Purpose |
|---|---|
| [langpivot/costmodel.py](langpivot/costmodel.py) | Pricing table + pivot-vs-direct arithmetic + closed-form break-even |
| [langpivot/detect.py](langpivot/detect.py) | Free script-based language detection (skip rule) |
| [langpivot/predictor.py](langpivot/predictor.py) | EN-token predictor (least squares, no dependencies) |
| [langpivot/router.py](langpivot/router.py) | The routing decision with full cost breakdown |
| [langpivot/middleware.py](langpivot/middleware.py) | `PivotClient` — drop-in chat wrapper (OpenAI-compatible APIs) |
| [langpivot/corpus.py](langpivot/corpus.py) | Tatoeba parallel-pair builder |
| [scripts/run_study.py](scripts/run_study.py) | Full empirical study → `results.json` + trained predictor |

## Setup

```bash
pip install tiktoken   # only dependency (token counting for OpenAI models)
```

Data for the study (place in `data/`):

```bash
curl -LO https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2
curl -LO https://downloads.tatoeba.org/exports/per_language/ara/ara_sentences.tsv.bz2
curl -LO https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
curl -LO https://downloads.tatoeba.org/exports/per_language/jpn/jpn-eng_links.tsv.bz2
curl -LO https://downloads.tatoeba.org/exports/per_language/ara/ara-eng_links.tsv.bz2
```

## Reproduce the study

```bash
python -X utf8 scripts/run_study.py    # writes results.json + predictor_weights.json
python -X utf8 -m unittest discover -s tests
```

## Use the router / middleware

```python
from langpivot import EnTokenPredictor, PivotRouter, PivotClient

predictor = EnTokenPredictor.load("predictor_weights.json")
router = PivotRouter(
    main_model="gpt-4o",
    translator_model="gpt-4o-mini",
    predictor=predictor,
    r_out={"ja": 1.57, "ar": 1.30},   # measured output-side inflation (o200k)
)

# Decision only (no API call, no key needed):
d = router.decide("現在の大規模言語モデルは日本語の処理コストが高すぎます。" * 30)
print(d.route, d.reason)              # e.g. "pivot  pivot saves 24.1%"

# Full middleware (needs OPENAI_API_KEY; dry-runs without one):
client = PivotClient(router)
result = client.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "気候変動が日本の農業に与える影響を詳しく説明してください。"},
])
print(result["decision"].route, result["response"])
```

For Claude models, pass `count_tokens_fn=` backed by Anthropic's
`count_tokens` API (Claude's tokenizer is not public — tiktoken counts are
wrong for it).

## Security & privacy

- **The routing decision is fully local.** `PivotRouter.decide()`,
  `default_predictor()`, `detect_language()`, and the cost model make **no
  network calls** — they run offline on your text and never transmit it
  anywhere. The bundled model is plain JSON (no `pickle`, no `eval`/`exec`, no
  subprocess); the core package has **zero required dependencies**.
- **Only `PivotClient.chat()` talks to the outside world.** It calls the LLM
  API you configure, over TLS. API keys are read from environment variables
  (never hard-coded), are **never written to logs or error messages**, and the
  client **refuses to send a key over a non-`https://` endpoint** (raises
  `ValueError`).
- Report vulnerabilities privately — see [SECURITY.md](SECURITY.md).

## Honest limitations

- **Quality is unmeasured.** Pivoting adds two machine translations; register,
  cultural nuance, and intended ambiguity are at risk. See REPORT.md §4 for
  the evaluation protocol before any production use.
- Ratios come from Tatoeba (short, conversational). Domain text (legal,
  medical) should be re-measured with `scripts/run_study.py` on your own data.
- Prices drift; the table in `costmodel.py` is config, verified 2026-07-04.

---
MIT licensed. Created by Bashar Kellawi — BasharKilawe@gmail.com
