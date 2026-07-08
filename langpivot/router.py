"""The cost-aware routing decision: direct vs pivot-through-English.

For each request the router:
  1. detects the source language (script-based, free);
  2. if already English -> direct, no further work (the user's 'skip' rule);
  3. counts the actual source-language tokens with the main model's tokenizer;
  4. predicts the English token count with the trained per-language predictor
     (no translation performed);
  5. compares direct vs pivot cost under the pricing table and returns the
     cheaper route with the full cost breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .costmodel import CostBreakdown, CostModel, ModelPrice, PRICES
from .detect import detect_language
from .predictor import EnTokenPredictor


@dataclass
class RouteDecision:
    route: str                    # "direct" | "pivot"
    language: str
    src_tokens_in: int
    est_en_tokens_in: float
    est_en_tokens_out: float
    est_inflation_in: float
    costs: CostBreakdown | None = None
    reason: str = ""
    extras: dict = field(default_factory=dict)


class PivotRouter:
    def __init__(
        self,
        main_model: str = "gpt-4o",
        translator_model: str = "gpt-4o-mini",
        predictor: EnTokenPredictor | None = None,
        r_out: dict[str, float] | None = None,
        margin: float = 1.05,
        count_tokens_fn=None,
    ):
        """``margin``: pivot only when direct_cost > margin * pivot_cost, so
        borderline cases stay direct (avoids quality/latency risk for ~0 gain).

        ``r_out``: output-side inflation ratio per language (tokens the answer
        costs in the source language per English token of the same answer),
        measured from parallel data by the study script.

        ``count_tokens_fn(text) -> int`` overrides the tokenizer (e.g. use the
        Anthropic count_tokens API for Claude models — their tokenizer is not
        public and tiktoken must not be used for them).
        """
        self.main = PRICES[main_model]
        self.translator = PRICES[translator_model]
        self.cost_model = CostModel(self.main, self.translator)
        self.predictor = predictor or EnTokenPredictor()
        self.r_out = r_out or {}
        self.margin = margin
        if count_tokens_fn is not None:
            self._count = count_tokens_fn
        else:
            if not self.main.tokenizer:
                raise ValueError(
                    f"{self.main.name} has no public tokenizer; pass count_tokens_fn"
                )
            import tiktoken

            enc = tiktoken.get_encoding(self.main.tokenizer)
            self._count = lambda text: len(enc.encode(text))

    def decide(self, text: str, expected_en_tokens_out: float = 300.0) -> RouteDecision:
        lang = detect_language(text)
        if lang not in ("ja", "ar"):
            return RouteDecision(
                route="direct", language=lang, src_tokens_in=0,
                est_en_tokens_in=0, est_en_tokens_out=expected_en_tokens_out,
                est_inflation_in=1.0,
                reason="already in pivot language or unsupported language",
            )
        src_tokens = self._count(text)
        try:
            en_tokens = self.predictor.predict(lang, text)
        except KeyError:
            return RouteDecision(
                route="direct", language=lang, src_tokens_in=src_tokens,
                est_en_tokens_in=src_tokens, est_en_tokens_out=expected_en_tokens_out,
                est_inflation_in=1.0,
                reason=f"no predictor for {lang}; defaulting to direct",
            )
        r_in = max(1.0, src_tokens / en_tokens)
        r_out = self.r_out.get(lang, r_in)
        costs = self.cost_model.pivot_cost(
            t_en_in=en_tokens, t_en_out=expected_en_tokens_out,
            r_in=r_in, r_out=r_out,
        )
        pivot_wins = costs.direct > self.margin * costs.pivot
        return RouteDecision(
            route="pivot" if pivot_wins else "direct",
            language=lang,
            src_tokens_in=src_tokens,
            est_en_tokens_in=en_tokens,
            est_en_tokens_out=expected_en_tokens_out,
            est_inflation_in=r_in,
            costs=costs,
            reason=(
                f"pivot saves {costs.savings_pct:.1f}%"
                if pivot_wins
                else f"pivot would change cost by {-costs.savings_pct:.1f}% (margin {self.margin})"
            ),
        )
