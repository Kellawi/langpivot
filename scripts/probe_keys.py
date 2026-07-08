# -*- coding: utf-8 -*-
"""Probe the three provider keys with a tiny live call each (verifies quota,
not just presence). Prints per-provider OK/error only — never the key."""
import io, json, os, sys, urllib.request, urllib.error
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())

def try_(name, fn):
    try:
        out = fn(); print(f"{name}: OK  ->  {out}")
    except urllib.error.HTTPError as e:
        print(f"{name}: HTTP {e.code}  {e.read().decode('utf-8','replace')[:140]}")
    except Exception as e:
        print(f"{name}: ERR {type(e).__name__}: {str(e)[:140]}")

# OpenAI
ok = os.environ.get("OPENAI_API_KEY")
if ok:
    try_("OpenAI/gpt-4o-mini", lambda: post(
        "https://api.openai.com/v1/chat/completions",
        {"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}],"max_tokens":5},
        {"Content-Type":"application/json","Authorization":f"Bearer {ok}"}
    )["choices"][0]["message"]["content"][:20])
else:
    print("OpenAI: no key")

# Anthropic
ak = os.environ.get("ANTHROPIC_API_KEY")
if ak:
    try_("Anthropic/claude-haiku-4-5", lambda: "".join(
        b.get("text","") for b in post(
        "https://api.anthropic.com/v1/messages",
        {"model":"claude-haiku-4-5","max_tokens":10,
         "messages":[{"role":"user","content":"ping"}]},
        {"Content-Type":"application/json","x-api-key":ak,"anthropic-version":"2023-06-01"}
    )["content"])[:20])
else:
    print("Anthropic: no key")

# Gemini — list available models first, then tiny generate
gk = os.environ.get("GEMINI_API_KEY")
if gk:
    try:
        models = get(f"https://generativelanguage.googleapis.com/v1beta/models?key={gk}")
        names = [m["name"].split("/")[-1] for m in models.get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
        flash = [n for n in names if "flash" in n and "lite" not in n]
        pro = [n for n in names if "pro" in n]
        print(f"Gemini: {len(names)} gen models. flash={flash[:4]} pro={pro[:4]}")
        pick = (flash or names)[0]
        txt = post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{pick}:generateContent?key={gk}",
            {"contents":[{"parts":[{"text":"ping"}]}],
             "generationConfig":{"maxOutputTokens":10}},
            {"Content-Type":"application/json"})
        out = txt["candidates"][0]["content"]["parts"][0]["text"][:20]
        print(f"Gemini/{pick}: OK -> {out!r}")
    except urllib.error.HTTPError as e:
        print(f"Gemini: HTTP {e.code}  {e.read().decode('utf-8','replace')[:140]}")
    except Exception as e:
        print(f"Gemini: ERR {type(e).__name__}: {str(e)[:140]}")
else:
    print("Gemini: no key")
