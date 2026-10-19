"""Deterministic long documents for the long-context and agentic modes, sized in tokens of the served model.

The text is generated from fixed seeds by the same generators as the prompt set's long-context documents
(tools/build_prompts.py: fictional minutes, logs, code, sales data, support chats, requirements, a chronicle,
configuration, diffs), concatenated as numbered sections. A salt line at the very start makes a prompt unique
so that no prefix cache can serve it (cold); reusing a document with another question exercises the cache.
"""
import importlib.util
import os
import random

from . import client

PKG = os.path.dirname(os.path.abspath(__file__))
_BP = None

QUESTIONS = [
    "List every decision, owner and open risk mentioned anywhere in the document, grouped by section type.",
    "Write a concise executive summary of the document, then list the five most surprising details in it.",
    "Find inconsistencies between sections of the document and explain each one briefly.",
    "Which sections would you read first to plan next week's work, and why? Then draft that plan.",
    "Summarise the incidents, anomalies and errors that appear in the document, with their likely causes.",
    "Write a timeline of the events in the document, then name the three people who appear most often.",
]


def _builder():
    global _BP
    if _BP is None:
        path = os.path.join(PKG, "..", "tools", "build_prompts.py")
        spec = importlib.util.spec_from_file_location("speedbench_build_prompts", path)
        _BP = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BP)
    return _BP


def document(n_chars, seed=0, salt=None):
    """~n_chars of generated text (deterministic for a seed); salt goes on the first line"""
    bp = _builder()
    r = random.Random(7919 * seed + 17)
    out = ["Document reference: %s\n\n" % salt] if salt is not None else []
    size = sum(map(len, out))
    i = 0
    while size < n_chars:
        name, fn, _ = bp.LONG[(i + seed) % len(bp.LONG)]
        sec = "\n### Section %d (%s)\n\n%s" % (i + 1, name, fn(r))
        out.append(sec)
        size += len(sec)
        i += 1
    text = "".join(out)
    cut = text.rfind("\n", 0, n_chars)
    return text[:cut + 1] if cut > n_chars // 2 else text[:n_chars]


def messages(doc, question):
    return [{"role": "user", "content": doc + "\n\n---\n\n" + question}]


def count_tokens(url, model, text, headers=None):
    """prompt tokens of one user message via /tokenize (vLLM, gr8); None if unavailable"""
    for body in ({"model": model, "messages": [{"role": "user", "content": text}]}, {"model": model, "prompt": text}):
        try:
            j, _ = client.http_json(url.rstrip("/") + "/tokenize", body, timeout=60, headers=headers)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(j, dict) and isinstance(j.get("count"), int):
            return j["count"]
        if isinstance(j, dict) and isinstance(j.get("tokens"), list):
            return len(j["tokens"])
    return None


def calibrate(url, model, headers=None, chat_tokens=None):
    """characters per token of generated documents for this model. chat_tokens(messages) -> prompt tokens (a
    max_tokens=1 request) is the fallback when /tokenize is not served. Returns (chars_per_token, method)."""
    sample = document(24000, seed=99)
    small = document(2000, seed=99)
    n_big, n_small = count_tokens(url, model, sample, headers), count_tokens(url, model, small, headers)
    method = "tokenize"
    if not n_big or not n_small:
        if chat_tokens is None:
            return 3.6, "assumed"
        n_big, n_small = chat_tokens(messages(sample, "x")), chat_tokens(messages(small, "x"))
        method = "chat usage"
    if not n_big or not n_small or n_big <= n_small:
        return 3.6, "assumed"
    return (len(sample) - len(small)) / (n_big - n_small), method


def sized(url, model, target_tokens, cpt, seed=0, salt=None, question=QUESTIONS[0], headers=None, tol=0.01, tries=6):
    """a document whose chat prompt (document + question) has ~target_tokens tokens. Sections differ in token
    density, so one characters-per-token ratio misses by up to ~10-20 %: with /tokenize every candidate is counted
    and the length is searched (secant steps inside a bracket) until the count is within tol. Returns
    (text, its counted tokens or None if /tokenize is unavailable, converged); the count always belongs to the
    returned text."""
    base = len(question) + 16
    n_chars = max(200, int(target_tokens * cpt) - base)
    lo = hi = None                                    # (chars, tokens) below / above the target
    best = None
    for _ in range(tries):
        doc = document(n_chars, seed=seed, salt=salt)
        n = count_tokens(url, model, messages(doc, question)[0]["content"], headers)
        if not n:
            return doc, None, False
        if best is None or abs(n - target_tokens) < abs(best[1] - target_tokens):
            best = (doc, n)
        if abs(n / target_tokens - 1) <= tol:
            return doc, n, True
        if n < target_tokens:
            lo = (len(doc), n) if lo is None or len(doc) > lo[0] else lo
        else:
            hi = (len(doc), n) if hi is None or len(doc) < hi[0] else hi
        if lo and hi and hi[1] > lo[1]:               # interpolate inside the bracket
            n_chars = int(lo[0] + (hi[0] - lo[0]) * (target_tokens - lo[1]) / (hi[1] - lo[1]))
        else:
            n_chars = max(200, int(len(doc) * target_tokens / n))
    return best[0], best[1], False
