"""PivotClient: drop-in chat wrapper implementing adaptive language pivoting.

Flow per request (the user-described architecture, plus the cost gate):

    messages (JA/AR/EN) --> PivotRouter.decide()
        direct: main model sees the original messages
        pivot : translator model translates system+user messages to English,
                main model answers in English,
                translator translates the answer back to the source language

Translation uses the cheap translator model itself (one call per direction,
all messages batched into one translation request). If no API key is
available the client runs in dry-run mode: it still routes and reports
projected costs, but returns the decision instead of calling any API.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .router import PivotRouter, RouteDecision

_TRANSLATE_SYSTEM = (
    "You are a professional translator. Translate the user content into {target}. "
    "Preserve meaning, tone, formality level, formatting, code blocks, and "
    "placeholders exactly. Output ONLY the translation, no commentary."
)


class PivotClient:
    def __init__(
        self,
        router: PivotRouter,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        main_model: str | None = None,
        translator_model: str | None = None,
        dry_run: bool = False,
    ):
        self.router = router
        self.dry_run = dry_run
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # Never transmit a credential over a plaintext channel. Note: the error
        # deliberately reports only the URL scheme, never the key itself.
        if self.api_key and not self.base_url.startswith("https://"):
            raise ValueError(
                "PivotClient refuses to send an API key over a non-HTTPS "
                f"base_url (scheme {self.base_url.split('://', 1)[0]!r}); "
                "use an https:// endpoint."
            )
        self.main_model = main_model or router.main.name
        self.translator_model = translator_model or router.translator.name

    # ------------------------------------------------------------------ http

    def _chat(self, model: str, messages: list[dict], max_tokens: int | None = None) -> dict:
        body = {"model": model, "messages": messages}
        if max_tokens:
            body["max_tokens"] = max_tokens
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))

    def _translate(self, text: str, target: str) -> str:
        resp = self._chat(
            self.translator_model,
            [
                {"role": "system", "content": _TRANSLATE_SYSTEM.format(target=target)},
                {"role": "user", "content": text},
            ],
        )
        return resp["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ api

    def chat(
        self,
        messages: list[dict],
        expected_en_tokens_out: float = 300.0,
        max_tokens: int | None = None,
    ) -> dict:
        """Returns {"decision": RouteDecision, "response": str | None,
        "dry_run": bool, "usage": [...raw usage dicts...]}."""
        combined = "\n".join(m.get("content", "") for m in messages)
        decision: RouteDecision = self.router.decide(combined, expected_en_tokens_out)

        if self.api_key is None or self.dry_run:
            return {"decision": decision, "response": None, "dry_run": True, "usage": []}

        usage = []
        if decision.route == "direct":
            resp = self._chat(self.main_model, messages, max_tokens)
            usage.append(resp.get("usage", {}))
            return {
                "decision": decision,
                "response": resp["choices"][0]["message"]["content"],
                "dry_run": False,
                "usage": usage,
            }

        # pivot: translate every non-English message to English
        lang_names = {"ja": "Japanese", "ar": "Arabic"}
        src_lang = lang_names.get(decision.language, decision.language)
        en_messages = []
        for m in messages:
            content = m.get("content", "")
            from .detect import detect_language

            if detect_language(content) in ("ja", "ar"):
                content = self._translate(content, "English")
            en_messages.append({**m, "content": content})

        resp = self._chat(self.main_model, en_messages, max_tokens)
        usage.append(resp.get("usage", {}))
        en_answer = resp["choices"][0]["message"]["content"]
        final_answer = self._translate(en_answer, src_lang)
        return {
            "decision": decision,
            "response": final_answer,
            "dry_run": False,
            "usage": usage,
        }
