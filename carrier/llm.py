"""Optional open-weight LLM over an OpenAI-compatible endpoint (e.g. llama.cpp llama-server with Qwen2.5).

Set LLM_BASE_URL (and optionally LLM_MODEL, LLM_API_KEY) to enable. Without it the agent uses
rules-based NLU and the evaluation uses rule checks plus precomputed LLM-judge results.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

ACTS = ["ask_bill", "ask_data", "ask_roaming", "ask_upgrade", "request_credit", "request_human",
        "request_cancel", "give_identity", "express_frustration", "thank_bye", "confirm", "other"]


def _cfg(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def available() -> bool:
    return bool(_cfg("LLM_BASE_URL"))


def chat(messages: list[dict], max_tokens: int = 200, temperature: float = 0.0, timeout: float = 600) -> str:
    url = _cfg("LLM_BASE_URL").rstrip("/") + "/chat/completions"
    body = {"model": _cfg("LLM_MODEL", "qwen2.5"), "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    key = _cfg("LLM_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def classify_act(text: str) -> str | None:
    try:
        out = chat([{"role": "system", "content": "Classify the caller's utterance in a cell phone carrier support call. "
                     "Answer with exactly one label from: " + ", ".join(ACTS)},
                    {"role": "user", "content": text}], max_tokens=8, timeout=30)
    except Exception:
        return None
    m = re.search("|".join(ACTS), out)
    return m.group(0) if m else None


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
