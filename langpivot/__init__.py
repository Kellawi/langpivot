"""Adaptive Language Pivoting for Reducing LLM API Cost.

A cost-aware routing layer that decides, per request, whether to send
Japanese/Arabic text directly to an LLM API or to pivot it through English
(translate in -> run in English -> translate out) — like a database query
optimizer, but for tokens.

Direct cost : P_in * r_in * T_en_in            + P_out * r_out * T_en_out
Pivot cost  : (translate-in) + (main call in EN) + (translate-out)

Pivoting wins only when the token inflation ratio r of the source language
and the price gap between the main model and the translator are both large
enough. The router estimates the English token count *without* translating,
using a linear predictor trained on real parallel corpora.
"""

from .costmodel import PRICES, CostModel, ModelPrice
from .detect import detect_language
from .predictor import EnTokenPredictor, default_predictor
from .router import PivotRouter, RouteDecision
from .middleware import PivotClient

__version__ = "1.0.0"
__all__ = [
    "PRICES", "CostModel", "ModelPrice", "detect_language",
    "EnTokenPredictor", "default_predictor",
    "PivotRouter", "RouteDecision", "PivotClient",
]
