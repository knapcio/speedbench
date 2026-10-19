"""Quality guard: did a speed lever break the outputs? A guard against gross breakage, not a quality benchmark.

Two opt-in checks (`run --quality-guard`):

1. Exact-answer items (prompts/speedbench-guard-v1.json, 30 short arithmetic, factual, format, code and translation
   questions with deterministic graders), run at T=0 in every arm after the timed workload. `compare` reports the
   pass counts per arm and flags a drop (paired per item, exact McNemar test). Thirty easy items detect a broken
   template, kernel or sampler; they cannot measure small quality differences.
2. Divergence of the T=0 outputs (needs the generated text: `--quality-guard` implies `--keep-text`): for prompts
   whose output differs between the arms, the position of the first differing token. A numerics change usually
   lets two outputs agree for a while and then flip at a near-tie; outputs that differ from the first few tokens
   in most prompts point to a change of the prompt, template or sampling instead.
"""
import json
import math
import os
import re
import statistics

from . import prompts as P

DEFAULT_SET = os.path.join(P.PKG, "..", "prompts", "speedbench-guard-v1.json")


def load(path=None):
    with open(path or DEFAULT_SET, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------- grading
def _strip(s):
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s).strip()
    return s


def _last_line(s):
    lines = [x.strip() for x in s.strip().splitlines() if x.strip()]
    return lines[-1] if lines else ""


def _norm(s, case=False):
    s = s.replace("*", "").replace("`", "").strip().strip("\"'.!;:").strip()
    s = re.sub(r"\s+", " ", s)
    return s if case else s.lower()


def grade(grader, content):
    """(passed, extracted answer) for one reply's content"""
    content = content or ""
    kind = grader["type"]
    if kind == "number":
        txt = re.sub(r"(?<=\d),(?=\d{3}\b)", "", content)
        nums = re.findall(r"-?\d+(?:\.\d+)?", txt)
        if not nums:
            return False, None
        v = float(nums[-1])
        return abs(v - float(grader["value"])) < 1e-9, nums[-1]
    if kind == "text":
        case = grader.get("case", False)
        line = _norm(_last_line(_strip(content)), case)
        for acc in grader["accept"]:
            a = acc if case else acc.lower()
            if line == a or re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(a), line):
                return True, line[:80]
        return False, line[:80]
    if kind == "json":
        s = _strip(content)
        cand = [s]
        m = re.search(r"[\[{].*[\]}]", s, re.S)
        if m:
            cand.append(m.group(0))
        for c in cand:
            try:
                v = json.loads(c)
            except ValueError:
                continue
            return v == grader["value"], c[:80]
        return False, s[:80]
    raise ValueError("unknown grader %r" % kind)


def run_items(arms, a, one, log=None):
    """send every guard item at T=0 to every arm (alternating order per item); {arm label: result}"""
    body = load(getattr(a, "guard_set", None))
    t_saved = a.temperature
    a.temperature = 0.0
    out = {arm.label: [] for arm in arms}
    try:
        for n, item in enumerate(body["items"]):
            order = arms if n % 2 == 0 else arms[::-1]
            for arm in order:
                r = one(arm, item, a, a.guard_max_tokens)
                text = (r.get("text") or {}) if r.get("ok") else {}
                content = text.get("content") or ""
                ok, ans = grade(item["grader"], content) if r.get("ok") else (False, None)
                rec = {"id": item["id"], "pass": bool(ok), "answer": ans, "ok": r.get("ok", False),
                       "finish_reason": r.get("finish_reason"), "completion_tokens": r.get("completion_tokens"),
                       "sha256": r.get("sha256"), "content": content[-300:], "error": r.get("error")}
                if r.get("ok") and not content.strip() and r.get("finish_reason") == "length":
                    rec["note"] = "no answer within max_tokens (still reasoning)"
                out[arm.label].append(rec)
                if log:
                    log(arm, rec)
    finally:
        a.temperature = t_saved
    return {lab: {"version": body.get("version"), "max_tokens": a.guard_max_tokens, "items": recs,
                  "passed": sum(r["pass"] for r in recs), "total": len(recs),
                  "truncated": sum(1 for r in recs if r.get("note"))} for lab, recs in out.items()}


def mcnemar_one_sided(b, c):
    """P(at least b of b+c discordant items go against B | no difference): exact binomial, p = 0.5"""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def compare_guard(ga, gb):
    """paired comparison of two arms' guard results"""
    pa = {r["id"]: r for r in ga["items"]}
    pb = {r["id"]: r for r in gb["items"]}
    ids = [i for i in pa if i in pb]
    lost = [i for i in ids if pa[i]["pass"] and not pb[i]["pass"]]
    won = [i for i in ids if pb[i]["pass"] and not pa[i]["pass"]]
    same_text = sum(1 for i in ids if pa[i].get("sha256") and pa[i]["sha256"] == pb[i].get("sha256"))
    p = mcnemar_one_sided(len(lost), len(won))
    drop = len(lost) > len(won) and (p < 0.10 or len(lost) - len(won) >= 3)
    return {"n": len(ids), "passed_A": sum(pa[i]["pass"] for i in ids), "passed_B": sum(pb[i]["pass"] for i in ids),
            "lost_in_B": lost, "won_in_B": won, "p_one_sided": p, "drop": drop, "identical_outputs": same_text,
            "truncated_A": ga.get("truncated", 0), "truncated_B": gb.get("truncated", 0)}


# ------------------------------------------------------------------------------------------- divergence
def _stream(r):
    t = r.get("text") or {}
    tools = t.get("tool_calls") or []
    # tool calls: the function name and its arguments (a changed function name is a divergence too)
    return (t.get("reasoning") or "") + (t.get("content") or "") + "".join(
        (x.get("name") or "") + (x.get("arguments") or "") for x in tools)


def first_divergence(ra, rb):
    """approximate token position where two outputs first differ (None if identical or text missing). The
    character position is mapped to tokens with A's chunk trace (cumulative tokens per chunk when the server
    reports them), otherwise with A's average characters per token."""
    if not ra.get("text") or not rb.get("text"):
        return None
    sa, sb = _stream(ra), _stream(rb)
    if sa == sb:
        return None
    p = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x != y), min(len(sa), len(sb)))
    ch = ra.get("chunks") or []
    if ch and all(c[2] >= 0 for c in ch):
        chars, prev = 0, 0
        for _, n, cum, _ in ch:
            if chars + n > p:
                return prev
            chars += n
            prev = cum
        return prev
    n_tok = ra.get("completion_tokens") or 0
    return int(p * n_tok / len(sa)) if sa else 0


def divergence(As, Bs, measured, early=(16, 64)):
    """first-divergence summary over prompts whose (single) T=0 output differs between the arms"""
    def first(docs):
        out = {}
        for d in docs:
            for r in measured(d):
                if r.get("ok") and r.get("text") and r["key"] not in out:
                    out[r["key"]] = r
        return out
    ra, rb = first(As), first(Bs)
    keys = [k for k in ra if k in rb]
    if not keys:
        return None
    pos, n_same = [], 0
    for k in keys:
        if ra[k]["sha256"] == rb[k]["sha256"]:
            n_same += 1
            continue
        d = first_divergence(ra[k], rb[k])
        if d is not None:
            pos.append(d)
    out = {"n": len(keys), "identical": n_same, "differing": len(keys) - n_same, "positions": sorted(pos)}
    if pos:
        q = statistics.quantiles(pos, n=4) if len(pos) >= 2 else [pos[0]] * 3
        out.update({"min": min(pos), "p25": q[0], "median": statistics.median(pos), "p75": q[2],
                    "early": {n: sum(x < n for x in pos) for n in early},
                    "median_output_tokens": statistics.median(ra[k]["completion_tokens"] for k in keys)})
    return out
