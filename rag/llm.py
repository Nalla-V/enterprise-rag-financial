"""Thin wrapper around any OpenAI-compatible chat endpoint
(vLLM server on ALICE, Databricks serving endpoint, or a hosted API)."""
import json
import re

from . import config

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


def chat(messages, max_tokens=512, temperature=0.0, model=None):
    r = client().chat.completions.create(model=model or config.LLM_MODEL, messages=messages,
                                         max_tokens=max_tokens, temperature=temperature)
    return r.choices[0].message.content.strip()


def ask(system, user, **kw):
    return chat([{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)


def ask_json(system, user, default=None, **kw):
    """Ask for JSON and parse it robustly (models sometimes wrap it in ``` fences)."""
    text = ask(system, user, **kw)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else default
    except json.JSONDecodeError:
        return default