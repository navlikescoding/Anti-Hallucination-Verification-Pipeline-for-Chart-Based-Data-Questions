"""
llm.py — asking the LLMs, with a disk cache.

Two providers, one interface:
  - Ollama  : open-source models running locally on your laptop (no key needed)
  - Gemini  : Google's commercial model, free tier (key read from an environment variable)

Only Python's built-in `urllib` is used for the web requests, so no extra libraries.

THE CACHE: every response is appended to cache/llm_responses.jsonl, keyed by a
fingerprint of (model + prompt + settings). Asking the same thing twice reads the
saved answer instead of calling the model again, so reruns are free and instant,
and an interrupted run continues where it stopped.
"""
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request

from . import config
from .config import TABLE_DIR

log = logging.getLogger("pipeline")


# ===================================================================== 1. prompts
SYSTEM_PROMPT = (
    "You answer questions about charts for blind and low-vision users. "
    "You are given the chart's underlying data as a CSV table. "
    "Answer using only that data."
)

STYLE_INSTRUCTIONS = {
    "short":    "Reply with ONLY the answer: a single number, name, or Yes/No. "
                "No units, no explanation.",
    "sentence": "Reply in ONE short, natural sentence that a screen reader could read aloud, "
                "stating the answer clearly.",
}


def read_table_text(table_file: str) -> str:
    """The raw CSV text exactly as in ChartQA (including %, $, footnote stars)."""
    return (TABLE_DIR / table_file).read_text(encoding="utf-8").strip()


def build_prompt(question: str, table_file: str, chart_type: str, style: str) -> str:
    """The user message sent to the LLM: chart spec + data + question + style rule."""
    return (
        f"Chart type: {chart_type} chart\n"
        f"Chart data (CSV):\n{read_table_text(table_file)}\n\n"
        f"Question: {question}\n\n"
        f"{STYLE_INSTRUCTIONS[style]}"
    )


def build_reprompt(question, table_file, chart_type, style, wrong_answer, correct_value) -> str:
    """Section 3: the correction prompt for a flagged answer. The model is shown its
    earlier answer and the verified value, and asked to restate the answer."""
    return (
        f"Chart type: {chart_type} chart\n"
        f"Chart data (CSV):\n{read_table_text(table_file)}\n\n"
        f"Question: {question}\n\n"
        f"Your previous answer was: {wrong_answer}\n"
        f"A verification check computed from the data shows the correct answer is: {correct_value}\n\n"
        f"Give the corrected answer. {STYLE_INSTRUCTIONS[style]}"
    )


# ===================================================================== 2. cache
def cache_key(model_cfg: dict, prompt: str, temperature: float, sample_idx: int = 0,
              system: str = None, images: list = None) -> str:
    """A unique fingerprint for one request. sample_idx lets self-consistency
    store several answers to the same prompt. Images (Section 3b) are included as a
    hash of their bytes; text-only requests get exactly the same key as before."""
    parts = [model_cfg["provider"], model_cfg["model"], system or SYSTEM_PROMPT,
             prompt, temperature, sample_idx]
    if images:
        parts.append([hashlib.sha256(img.encode("ascii")).hexdigest() for img in images])
    raw = json.dumps(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    """Loads all saved responses into memory; appends each new one to disk immediately."""

    def __init__(self, path=config.LLM_CACHE_FILE):
        self.path = path
        self.data = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec

    def get(self, key):
        return self.data.get(key)

    def put(self, rec: dict):
        self.data[rec["key"]] = rec
        with open(self.path, "a", encoding="utf-8") as f:   # append = never loses earlier answers
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ===================================================================== 3. providers
class QuotaExhausted(Exception):
    """Raised when the free tier refuses further requests (e.g. daily limit hit)."""


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_ollama(model: str, prompt: str, temperature: float, seed: int,
                 system: str = None, images: list = None, max_tokens: int = None) -> str:
    user_msg = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = images                       # base64-encoded PNGs
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system or SYSTEM_PROMPT}, user_msg],
        "stream": False,
        "options": {"temperature": temperature, "seed": seed,
                    "num_predict": max_tokens or config.MAX_OUTPUT_TOKENS},
    }
    # reading an image on a CPU can take minutes, so allow longer
    out = _post_json(f"{config.OLLAMA_URL}/api/chat", payload, {},
                     timeout=900 if images else 300)
    return out["message"]["content"]


def _call_gemini(model: str, prompt: str, temperature: float, seed: int,
                 system: str = None, images: list = None, max_tokens: int = None) -> str:
    key = os.environ.get(config.GEMINI_KEY_ENV)
    if not key:
        raise RuntimeError(f"Environment variable {config.GEMINI_KEY_ENV} is not set.")
    parts = [{"inline_data": {"mime_type": "image/png", "data": img}} for img in (images or [])]
    parts.append({"text": prompt})
    payload = {
        "systemInstruction": {"parts": [{"text": system or SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature, "seed": seed,
                             "maxOutputTokens": max_tokens or config.MAX_OUTPUT_TOKENS},
    }
    out = _post_json(f"{config.GEMINI_URL}/{model}:generateContent", payload,
                     {"x-goog-api-key": key})          # key goes in a header, never in the URL
    parts = out["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


PROVIDERS = {"ollama": _call_ollama, "gemini": _call_gemini}


def call_model(model_cfg: dict, prompt: str, temperature: float, seed: int,
               max_retries: int = 4, **extra) -> str:
    """Call a model, retrying on temporary errors (rate limits, server hiccups).
    `extra` may hold system=, images=, max_tokens= (used by Section 3b)."""
    fn = PROVIDERS[model_cfg["provider"]]
    for attempt in range(max_retries + 1):
        try:
            return fn(model_cfg["model"], prompt, temperature, seed, **extra)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")[:300]
            if e.code == 429 and attempt == max_retries:
                raise QuotaExhausted(body)
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = 2 ** (attempt + 2)                    # 4, 8, 16, 32 seconds
                print(f"  HTTP {e.code}, retrying in {wait}s ...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach {model_cfg['provider']}: {e.reason}")


# ===================================================================== 4. the one function notebooks use
def ask(model_cfg: dict, prompt: str, cache: ResponseCache,
        temperature: float = config.TEMPERATURE, sample_idx: int = 0,
        system: str = None, images: list = None, max_tokens: int = None) -> dict:
    """Return the model's answer for this prompt: from the cache if we have it,
    otherwise by calling the model (and saving the result).

    Optional (Section 3b): a different system prompt, base64 images, a longer
    output limit. Returns a record: {key, model, answer, latency_s, error, cached}.
    """
    key = cache_key(model_cfg, prompt, temperature, sample_idx, system, images)
    hit = cache.get(key)
    if hit is not None and hit.get("error") is None:
        return {**hit, "cached": True}

    t0 = time.time()
    answer, error = None, None
    try:
        answer = call_model(model_cfg, prompt, temperature, seed=config.SEED + sample_idx,
                            system=system, images=images, max_tokens=max_tokens).strip()
    except QuotaExhausted:
        raise                                           # stop the whole run; resume later
    except Exception as e:
        error = str(e)
        log.info(f"llm_error | {model_cfg['name']} | {error}")

    rec = {"key": key, "model": model_cfg["name"], "answer": answer,
           "latency_s": round(time.time() - t0, 2), "error": error}
    if error is None:
        cache.put(rec)                                  # only successful answers are cached
    if model_cfg["provider"] == "gemini":
        time.sleep(config.GEMINI_DELAY_S)               # be polite to the free tier
    return {**rec, "cached": False}


# ===================================================================== 5. health checks
def ollama_ready(model: str) -> bool:
    """True if Ollama is running and the model is downloaded."""
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=5) as r:
            names = [m["name"] for m in json.loads(r.read())["models"]]
    except Exception as e:
        print(f"Ollama not reachable at {config.OLLAMA_URL} ({e}). Is the Ollama app running?")
        return False
    if model not in names:
        print(f"Ollama is running but '{model}' isn't downloaded. Run:  ollama pull {model}")
        return False
    return True


def gemini_ready() -> bool:
    if not os.environ.get(config.GEMINI_KEY_ENV):
        print(f"{config.GEMINI_KEY_ENV} not set, so Gemini will be skipped.")
        return False
    return True
