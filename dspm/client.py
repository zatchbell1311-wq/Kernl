# dspm/client.py — LLMClient v3.2 (ported from DSPM v4.3 notebook)
import os
import re
import json
import time

from openai import OpenAI
from groq   import Groq

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MIN_OUTPUT_TOKENS_FLOOR = 800

CHAT_HINTS    = ("llama","qwen","gpt","gemma","mistral","kimi","deepseek",
                 "phi","instruct","chat","sonnet","nova","granite","command")
CHAT_EXCLUDES = ("guard","embed","whisper","tts","moderation","safety",
                 "flux","stable","diffusion","rerank","sora")


def _looks_like_chat(mid):
    m = mid.lower()
    return any(h in m for h in CHAT_HINTS) and not any(x in m for x in CHAT_EXCLUDES)


def _chat_score(mid):
    m = mid.lower(); s = 0
    for tag, pts in (("405b",40),("120b",30),("70b",30),("90b",30),("32b",20),
                     ("versatile",15),("instruct",10),("latest",5),("chat",5)):
        if tag in m: s += pts
    for tag in ("8b","9b","4b","3b","1b","mini","small","tiny","22m","86m"):
        if tag in m: s -= 15
    return s


class FatalBackendError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str, model: str, raw_clients: list):
        self.provider  = provider
        self.model     = model
        self._clients  = raw_clients
        self._idx      = 0
        self.n_calls   = 0

    def chat(self, max_retries: int = 5, **kwargs):
        kwargs.pop("model", None)
        if "max_tokens" in kwargs:
            kwargs["max_tokens"] = max(int(kwargs["max_tokens"]), MIN_OUTPUT_TOKENS_FLOOR)
        last_exc, backoff = None, [1, 2, 5, 15, 30]
        for attempt in range(max_retries):
            try:
                self.n_calls += 1
                resp    = self._clients[self._idx].chat.completions.create(
                    model=self.model, **kwargs
                )
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    kwargs["max_tokens"] = int(kwargs.get("max_tokens", MIN_OUTPUT_TOKENS_FLOOR)) * 2
                    continue
                return resp
            except Exception as e:
                last_exc = e
                status   = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404):
                    hint = {
                        404: f'MODEL "{self.model}" NOT FOUND.',
                        401: "API KEY INVALID — check your key.",
                        403: "KEY LACKS PERMISSION for this model.",
                    }.get(status, "Bad request — check parameters.")
                    raise FatalBackendError(f"HTTP {status}: {hint}") from e
                if len(self._clients) > 1:
                    self._idx = (self._idx + 1) % len(self._clients)
                if attempt < max_retries - 1:
                    time.sleep(backoff[min(attempt, 4)])
        if last_exc is not None:
            raise RuntimeError(f"LLM backend unreachable after {max_retries} attempts") from last_exc
        raise RuntimeError(f"{self.provider}/{self.model}: EMPTY content after {max_retries} attempts.")

    def ping(self) -> str:
        r = self.chat(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=256, temperature=0.0,
        )
        return (r.choices[0].message.content or "").strip()

    def probe_json(self):
        r = self.chat(
            messages=[
                {"role": "system", "content":
                 "Return ONLY a valid JSON array, nothing else. Schema: "
                 '[{"type":"entity","payload":"...","patch_id":"t_0","dependencies":[]}]'},
                {"role": "user", "content": "We standardised on PostgreSQL 15 with Redis caching."},
            ],
            temperature=0.0, max_tokens=1000,
        )
        raw = (r.choices[0].message.content or "").strip()
        jm  = re.search(r"\[.*\]", raw, re.DOTALL)
        if not jm:
            return None, raw
        try:
            return json.loads(jm.group()), raw
        except json.JSONDecodeError:
            return None, raw


def _resolve_provider(provider: str):
    or_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    g_keys = [(os.environ.get(f"GROQ_API_KEY{s}") or "").strip() for s in ("", "_2", "_3")]
    g_keys = [k for k in g_keys if k]
    if provider == "auto":
        if or_key:
            provider = "openrouter"
        elif g_keys:
            provider = "groq"
        else:
            raise FatalBackendError(
                "No API keys found. Set OPENROUTER_API_KEY or GROQ_API_KEY as environment variables."
            )
    if provider == "openrouter":
        if not or_key:
            raise FatalBackendError("PROVIDER=openrouter but OPENROUTER_API_KEY not set.")
        clients = [OpenAI(base_url=OPENROUTER_BASE_URL, api_key=or_key)]
    else:
        if not g_keys:
            raise FatalBackendError("PROVIDER=groq but no GROQ_API_KEY found.")
        clients = [Groq(api_key=k) for k in g_keys]
    return provider, clients


def _auto_pick_model(provider: str, client) -> str:
    from .engine import PATCH_TYPES  # avoid circular at module level
    preferred = {"openrouter": ["stealth/ox-alpha"], "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]}
    pref_list = preferred.get(provider, [])
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception:
        return pref_list[0] if pref_list else "llama-3.3-70b-versatile"
    for p in pref_list:
        if p in ids:
            return p
    chat_ids = [i for i in ids if _looks_like_chat(i)]
    if not chat_ids:
        raise FatalBackendError(f"No chat-capable models found on {provider}.")
    return sorted(chat_ids, key=_chat_score, reverse=True)[0]


def make_client(provider: str = "auto", model_override: str = None) -> LLMClient:
    provider, clients = _resolve_provider(provider)
    model  = model_override or _auto_pick_model(provider, clients[0])
    return LLMClient(provider, model, clients)
