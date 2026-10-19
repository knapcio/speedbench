"""Load the frozen prompt set and select the full or quick subset."""
import hashlib
import json
import os

PKG = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SET = os.path.join(PKG, "..", "prompts", "speedbench-v1.json")

# Warm-up requests: never measured, not part of the prompt set (original text).
WARMUP = [
    [{"role": "user", "content": "Count from one to twenty in words, one per line."}],
    [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}],
    [{"role": "user", "content": "Here is a list of fictional cities and their populations:\n" +
      "\n".join("City %d: %d inhabitants, founded in %d." % (i, 1000 + 37 * i * i, 1200 + 3 * i) for i in range(220)) +
      "\n\nWhich three cities are the largest?"}],
]


def load(path=None):
    path = path or DEFAULT_SET
    with open(path, "rb") as f:
        raw = f.read()
    body = json.loads(raw.decode("utf-8"))
    return body, hashlib.sha256(raw).hexdigest()


def select(body, mode="full", skip_tools=False, categories=None):
    out = []
    for p in body["prompts"]:
        if mode == "quick" and not p.get("quick"):
            continue
        if skip_tools and p.get("tools"):
            continue
        if categories and p["category"] not in categories:
            continue
        out.append(p)
    return out
