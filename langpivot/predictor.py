"""Predict the English token count of a text WITHOUT translating it.

This is the piece that makes the router a 'query optimizer': the pivot
cost depends on how many English tokens the translation would have, but
translating just to find out would already spend the money. Instead we fit,
per language, a tiny linear model

    t_en ~= w0 + w1 * n_chars + w2 * n_spaces + w3 * n_latin + w4 * n_digits

on real parallel pairs (least squares, closed form — no dependencies).
Features are chosen to be script-robust: Japanese length is measured in
chars; embedded Latin/digit runs translate roughly 1:1.
"""

from __future__ import annotations

import json


def _features(text: str) -> list[float]:
    n_latin = sum(1 for c in text if c.isascii() and c.isalpha())
    n_digits = sum(1 for c in text if c.isdigit())
    n_spaces = text.count(" ")
    return [1.0, float(len(text)), float(n_spaces), float(n_latin), float(n_digits)]


def _lstsq(X: list[list[float]], y: list[float]) -> list[float]:
    """Solve normal equations (X'X)w = X'y with Gaussian elimination."""
    k = len(X[0])
    xtx = [[sum(row[i] * row[j] for row in X) for j in range(k)] for i in range(k)]
    xty = [sum(row[i] * yi for row, yi in zip(X, y)) for i in range(k)]
    # ridge epsilon for numerical safety
    for i in range(k):
        xtx[i][i] += 1e-6
    # gaussian elimination with partial pivoting
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(xtx[r][col]))
        xtx[col], xtx[piv] = xtx[piv], xtx[col]
        xty[col], xty[piv] = xty[piv], xty[col]
        d = xtx[col][col]
        for r in range(col + 1, k):
            f = xtx[r][col] / d
            for c in range(col, k):
                xtx[r][c] -= f * xtx[col][c]
            xty[r] -= f * xty[col]
    w = [0.0] * k
    for i in range(k - 1, -1, -1):
        w[i] = (xty[i] - sum(xtx[i][j] * w[j] for j in range(i + 1, k))) / xtx[i][i]
    return w


class EnTokenPredictor:
    """Per-language linear predictor of English-translation token count."""

    def __init__(self, weights: dict[str, list[float]] | None = None):
        self.weights = weights or {}

    def fit(self, lang: str, texts: list[str], en_token_counts: list[int]) -> dict:
        X = [_features(t) for t in texts]
        y = [float(c) for c in en_token_counts]
        w = _lstsq(X, y)
        self.weights[lang] = w
        # goodness of fit
        preds = [max(1.0, sum(a * b for a, b in zip(row, w))) for row in X]
        mean_y = sum(y) / len(y)
        ss_res = sum((p - t) ** 2 for p, t in zip(preds, y))
        ss_tot = sum((t - mean_y) ** 2 for t in y) or 1.0
        mae = sum(abs(p - t) for p, t in zip(preds, y)) / len(y)
        mape = 100.0 * sum(abs(p - t) / t for p, t in zip(preds, y) if t) / len(y)
        return {"r2": 1 - ss_res / ss_tot, "mae": mae, "mape_pct": mape, "n": len(y)}

    def predict(self, lang: str, text: str) -> float:
        w = self.weights.get(lang)
        if w is None:
            raise KeyError(f"no predictor trained for language {lang!r}")
        return max(1.0, sum(a * b for a, b in zip(_features(text), w)))

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.weights, f)

    @classmethod
    def load(cls, path: str) -> "EnTokenPredictor":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


def default_predictor() -> "EnTokenPredictor":
    """Return a predictor pre-loaded with the weights bundled in the package.

    The weights (Japanese and Arabic, trained on Tatoeba parallel data) ship as
    package data and are read via ``importlib.resources``, so this works from an
    installed wheel with no files on disk and no network access.
    """
    from importlib.resources import files

    raw = (files("langpivot") / "data" / "predictor_weights.json").read_text(
        encoding="utf-8"
    )
    return EnTokenPredictor(json.loads(raw))
