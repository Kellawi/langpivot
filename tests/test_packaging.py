# -*- coding: utf-8 -*-
"""Tests for the bundled default predictor and the middleware HTTPS guard."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langpivot import (
    EnTokenPredictor,
    PivotClient,
    PivotRouter,
    default_predictor,
)


def _router():
    # count_tokens_fn keeps these tests independent of tiktoken.
    return PivotRouter(
        predictor=default_predictor(),
        count_tokens_fn=lambda s: max(1, len(s)),
    )


class TestDefaultPredictor(unittest.TestCase):
    def test_loads_bundled_weights(self):
        p = default_predictor()
        self.assertIsInstance(p, EnTokenPredictor)
        self.assertIn("ja", p.weights)
        self.assertIn("ar", p.weights)

    def test_predicts_positive(self):
        p = default_predictor()
        self.assertGreater(p.predict("ja", "私は毎日日本語を勉強しています。"), 0)
        self.assertGreater(p.predict("ar", "أريد أن أتعلم اللغة العربية بسرعة."), 0)

    def test_usable_in_router_returns_a_route(self):
        d = _router().decide("現在の大規模言語モデルは処理コストが高い。" * 20)
        self.assertEqual(d.language, "ja")
        self.assertIn(d.route, ("pivot", "direct"))


class TestHttpsGuard(unittest.TestCase):
    def setUp(self):
        # Make the guard tests hermetic w.r.t. a real key in the environment.
        self._saved = os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["OPENAI_API_KEY"] = self._saved

    def test_http_with_key_raises(self):
        with self.assertRaises(ValueError):
            PivotClient(_router(), api_key="sk-test-not-a-real-key",
                        base_url="http://insecure.example/v1")

    def test_error_message_omits_the_key(self):
        try:
            PivotClient(_router(), api_key="sk-secret-value",
                        base_url="http://insecure.example/v1")
            self.fail("expected ValueError")
        except ValueError as e:
            self.assertNotIn("sk-secret-value", str(e))

    def test_https_with_key_ok(self):
        c = PivotClient(_router(), api_key="sk-test-not-a-real-key",
                        base_url="https://api.openai.com/v1")
        self.assertEqual(c.api_key, "sk-test-not-a-real-key")

    def test_no_key_over_http_is_allowed(self):
        # No credential to leak -> guard does not fire (local/dry-run use).
        c = PivotClient(_router(), base_url="http://localhost:8080/v1")
        self.assertIsNone(c.api_key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
