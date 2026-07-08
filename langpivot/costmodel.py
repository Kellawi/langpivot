"""Pricing table and pivot-vs-direct cost arithmetic.

Prices are USD per 1M tokens and are CONFIGURABLE — they change often.
Values below were last verified 2026-07-04 (Claude prices from Anthropic's
current price sheet; OpenAI prices as commonly published — always re-check
before relying on absolute dollar figures; the *ratios* drive the routing
decision, and conclusions are robust to small price drift).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    name: str
    input_per_m: float   # USD per 1M input tokens
    output_per_m: float  # USD per 1M output tokens
    tokenizer: str       # tiktoken encoding name, or "" if unknown/not public


PRICES: dict[str, ModelPrice] = {
    # OpenAI (as of 2026-07; verify at openai.com/api/pricing)
    "gpt-4o": ModelPrice("gpt-4o", 2.50, 10.00, "o200k_base"),
    "gpt-4o-mini": ModelPrice("gpt-4o-mini", 0.15, 0.60, "o200k_base"),
    "gpt-4-turbo": ModelPrice("gpt-4-turbo", 10.00, 30.00, "cl100k_base"),
    # Anthropic (price sheet cached 2026-06; Claude tokenizer is not public —
    # use the count_tokens API for real counts, never tiktoken)
    "claude-opus-4-8": ModelPrice("claude-opus-4-8", 5.00, 25.00, ""),
    "claude-sonnet-5": ModelPrice("claude-sonnet-5", 3.00, 15.00, ""),
    "claude-haiku-4-5": ModelPrice("claude-haiku-4-5", 1.00, 5.00, ""),
    # Free local MT (e.g. NLLB / Marian running on your own hardware)
    "local-mt": ModelPrice("local-mt", 0.0, 0.0, ""),
}


@dataclass
class CostBreakdown:
    direct: float
    pivot: float
    translate_in: float
    main_call: float
    translate_out: float

    @property
    def savings(self) -> float:
        return self.direct - self.pivot

    @property
    def savings_pct(self) -> float:
        return 100.0 * self.savings / self.direct if self.direct else 0.0


class CostModel:
    """Cost arithmetic for one request, in USD.

    Token quantities are expressed in ENGLISH tokens (t_en_in, t_en_out)
    plus the language's inflation ratios (r_in, r_out): the same content in
    the source language costs r * t_en tokens on the main model's tokenizer.
    """

    def __init__(self, main: ModelPrice, translator: ModelPrice):
        self.main = main
        self.translator = translator

    def direct_cost(self, t_en_in: float, t_en_out: float, r_in: float, r_out: float) -> float:
        return (
            self.main.input_per_m * r_in * t_en_in
            + self.main.output_per_m * r_out * t_en_out
        ) / 1e6

    def pivot_cost(self, t_en_in: float, t_en_out: float, r_in: float, r_out: float) -> CostBreakdown:
        p = self.translator
        m = self.main
        # translate in: source-language text in, English text out
        translate_in = (p.input_per_m * r_in * t_en_in + p.output_per_m * t_en_in) / 1e6
        # main call entirely in English
        main_call = (m.input_per_m * t_en_in + m.output_per_m * t_en_out) / 1e6
        # translate out: English answer in, source-language answer out
        translate_out = (p.input_per_m * t_en_out + p.output_per_m * r_out * t_en_out) / 1e6
        pivot = translate_in + main_call + translate_out
        return CostBreakdown(
            direct=self.direct_cost(t_en_in, t_en_out, r_in, r_out),
            pivot=pivot,
            translate_in=translate_in,
            main_call=main_call,
            translate_out=translate_out,
        )

    def breakeven_ratio(self, out_in_ratio: float = 0.5) -> float:
        """Smallest inflation ratio r (assumed equal for input and output)
        at which pivoting becomes cheaper, for a request whose English
        output length is ``out_in_ratio`` times its English input length.

        Solves pivot_cost == direct_cost for r via the linear form.
        """
        m, p, k = self.main, self.translator, out_in_ratio
        # direct(r) = (M_i + M_o k) r          [per EN-input token, /1e6 dropped]
        # pivot(r)  = p_i r + p_o + M_i + M_o k + p_i k + p_o k r
        a = m.input_per_m + m.output_per_m * k          # slope of direct
        b = p.input_per_m + p.output_per_m * k          # r-dependent pivot part
        c = p.output_per_m + m.input_per_m + m.output_per_m * k + p.input_per_m * k
        if a <= b:
            return float("inf")  # translator too expensive: pivot never wins
        return c / (a - b)
