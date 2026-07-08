# -*- coding: utf-8 -*-
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langpivot.costmodel import PRICES, CostModel
from langpivot.detect import detect_language
from langpivot.predictor import EnTokenPredictor
from langpivot.router import PivotRouter


class TestDetect(unittest.TestCase):
    def test_japanese(self):
        self.assertEqual(detect_language("私は毎日日本語を勉強しています。"), "ja")

    def test_arabic(self):
        self.assertEqual(detect_language("أريد أن أتعلم اللغة العربية بسرعة."), "ar")

    def test_english(self):
        self.assertEqual(detect_language("The quick brown fox jumps over the lazy dog."), "en")

    def test_chinese_not_japanese(self):
        # han without kana must not be claimed as Japanese
        self.assertNotEqual(detect_language("我们今天去北京大学图书馆看书。"), "ja")

    def test_mixed_ja_with_ascii(self):
        self.assertEqual(detect_language("Pythonで機械学習を勉強中です。"), "ja")


class TestCostModel(unittest.TestCase):
    def setUp(self):
        self.cm = CostModel(PRICES["gpt-4o"], PRICES["gpt-4o-mini"])

    def test_pivot_wins_at_high_inflation(self):
        c = self.cm.pivot_cost(t_en_in=1000, t_en_out=500, r_in=2.0, r_out=2.0)
        self.assertLess(c.pivot, c.direct)

    def test_pivot_loses_at_no_inflation(self):
        c = self.cm.pivot_cost(t_en_in=1000, t_en_out=500, r_in=1.0, r_out=1.0)
        self.assertGreater(c.pivot, c.direct)

    def test_same_model_translator_never_wins(self):
        cm = CostModel(PRICES["gpt-4o"], PRICES["gpt-4o"])
        self.assertEqual(cm.breakeven_ratio(0.5), float("inf"))
        c = cm.pivot_cost(1000, 500, r_in=3.0, r_out=3.0)
        self.assertGreater(c.pivot, c.direct)

    def test_free_translator_breakeven_is_one(self):
        cm = CostModel(PRICES["gpt-4o"], PRICES["local-mt"])
        self.assertAlmostEqual(cm.breakeven_ratio(0.5), 1.0, places=6)

    def test_breakeven_matches_cost_crossover(self):
        r_star = self.cm.breakeven_ratio(0.5)
        lo = self.cm.pivot_cost(1000, 500, r_star * 0.99, r_star * 0.99)
        hi = self.cm.pivot_cost(1000, 500, r_star * 1.01, r_star * 1.01)
        self.assertGreater(lo.pivot, lo.direct)   # just below r*: direct wins
        self.assertLess(hi.pivot, hi.direct)      # just above r*: pivot wins


class TestPredictorAndRouter(unittest.TestCase):
    def test_predictor_learns_linear_map(self):
        # synthetic: en tokens = 0.4 * chars, exactly learnable
        texts = [("あ" * n) for n in range(10, 200, 7)]
        y = [int(0.4 * len(t)) for t in texts]
        p = EnTokenPredictor()
        fit = p.fit("ja", texts, y)
        self.assertGreater(fit["r2"], 0.99)
        self.assertAlmostEqual(p.predict("ja", "あ" * 100) / 40.0, 1.0, delta=0.1)

    def test_router_skips_english(self):
        p = EnTokenPredictor({"ja": [0, 0.5, 0, 0, 0]})
        r = PivotRouter(predictor=p)
        d = r.decide("This is plain English text and should go direct.")
        self.assertEqual(d.route, "direct")
        self.assertIn("already", d.reason)

    def test_router_routes_japanese(self):
        # force high inflation: predictor says EN needs far fewer tokens
        p = EnTokenPredictor({"ja": [0.0, 0.25, 0.0, 0.0, 0.0]})
        r = PivotRouter(predictor=p, r_out={"ja": 2.0})
        text = "現在の大規模言語モデルは日本語の処理コストが高すぎます。" * 20
        d = r.decide(text, expected_en_tokens_out=300)
        self.assertEqual(d.language, "ja")
        self.assertIsNotNone(d.costs)
        self.assertEqual(d.route, "pivot")


if __name__ == "__main__":
    unittest.main(verbosity=2)
