"""Shared helpers: paths, jsonl IO, and a disk-cached, rate-limited Gemini client."""
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = DATA / "cache"
for p in (DATA, RESULTS, CACHE):
    p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class GeminiCache:
    """Append-only jsonl cache keyed by sha1(model, system, prompt, config)."""

    def __init__(self, name):
        self.path = CACHE / f"{name}.jsonl"
        self.lock = threading.Lock()
        self.data = {r["k"]: r["v"] for r in read_jsonl(self.path)}

    def get(self, k):
        return self.data.get(k)

    def put(self, k, v):
        with self.lock:
            self.data[k] = v
            with open(self.path, "a") as f:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


_client = None


def client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _safety():
    from google.genai import types
    cats = ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
    return [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in cats]


_limits = {}
_limits_lock = threading.Lock()
RPM = float(os.environ.get("EAG_RPM", "6"))  # free tier: 15 RPM/model shared by six sessions


def _wait_turn(model):
    """Per-model spacing so this process never exceeds RPM requests/minute to one model."""
    while True:
        with _limits_lock:
            now = time.time()
            nxt = _limits.get(model, 0)
            if now >= nxt:
                _limits[model] = now + 60.0 / RPM
                return
        time.sleep(max(0.05, nxt - now))


def gemini(model, prompt, cache, system=None, temperature=0.0, max_tokens=1024,
           json_mode=False, thinking="minimal", retries=60):
    """Returns dict(text, finish, blocked). Cached on disk; rate-limited per model; retries 429/5xx
    with backoff (sleeps out per-day quota exhaustion instead of failing)."""
    from google.genai import types
    key = hashlib.sha1(json.dumps([model, system, prompt, temperature, max_tokens,
                                   json_mode, thinking]).encode()).hexdigest()
    hit = cache.get(key)
    if hit is not None:
        return hit
    cfg = dict(temperature=temperature, max_output_tokens=max_tokens,
               safety_settings=_safety(), http_options=types.HttpOptions(timeout=180000))
    if system:
        cfg["system_instruction"] = system
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    if thinking:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
    delay = 10
    for attempt in range(retries):
        _wait_turn(model)
        try:
            r = client().models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg))
            blocked = None
            if r.prompt_feedback is not None and r.prompt_feedback.block_reason:
                blocked = str(r.prompt_feedback.block_reason)
            finish = None
            text = ""
            if r.candidates:
                c = r.candidates[0]
                finish = str(c.finish_reason) if c.finish_reason else None
                if c.content and c.content.parts:
                    text = "".join(p.text or "" for p in c.content.parts if not getattr(p, "thought", False))
            out = {"text": text, "finish": finish, "blocked": blocked}
            cache.put(key, out)
            return out
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "PerDay" in msg and os.environ.get("EAG_NO_WAIT"):
                return {"text": "", "finish": "ERROR", "blocked": None, "error": msg[:300]}
            if "PerDay" in msg:
                print(f"[{model}] daily quota exhausted; sleeping 20 min", flush=True)
                time.sleep(1200)
                continue
            transient = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504",
                                               "UNAVAILABLE", "DEADLINE", "timed out", "Timeout",
                                               "Connection", "overloaded", "INTERNAL"))
            if not transient or attempt == retries - 1:
                return {"text": "", "finish": "ERROR", "blocked": None, "error": msg[:300]}
            print(f"[{model}] retry {attempt}: {msg[:90]}", flush=True)
            m = re.search(r"retry in ([0-9.]+)s", msg)
            time.sleep(float(m.group(1)) + 2 if m else delay + random.random() * 3)
            delay = min(delay * 2, 300)


def pmap(fn, items, workers=3, desc=""):
    """Thread map (max 4 concurrent Gemini requests per RULES.md) with a progress bar."""
    from tqdm import tqdm
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc, mininterval=10))


def parse_json(text):
    """Parse a JSON object from model output; a bare top-level list is wrapped as {"items": [...]}."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("\n") + 1:] if "\n" in text else text
    for s_ch, e_ch in (("{", "}"), ("[", "]")):
        s, e = text.find(s_ch), text.rfind(e_ch)
        if s < 0 or e < 0:
            continue
        if s_ch == "{" and 0 <= text.find("[") < s:
            continue  # top-level list containing objects
        try:
            v = json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            continue
        return {"items": v} if isinstance(v, list) else v
    return None
