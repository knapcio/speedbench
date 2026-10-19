"""Multi-turn agent workload (`run --mode agent`): conversations whose every request extends the previous one.

The frozen set (prompts/speedbench-agent-v1.json) holds 8 conversations of 4-8 requests. Request t of a
conversation = the system message + the task + the first t-1 turns, where a turn is a FROZEN assistant reply (text,
often with a tool call) followed by the next tool result or user message. Every request after the first therefore
starts with the previous request's prompt, which the engine's prefix cache can serve; only the new suffix (the
frozen reply and the new message) must be prefilled.

Why frozen replies: if the model's own replies were fed back, every engine and every lever would produce a different
history, so the prompts (and their lengths) of later turns would differ between arms and could not be paired. With
the history frozen, request t is the same prompt in every arm and run; only the generation of the request itself is
measured. The cost: the engine's cache holds its own generated tokens, not the frozen reply, so the reply is part of
the new suffix here, while a real agent loop may find it cached (engines that cache generated tokens).

Cache isolation (`--isolate`): an arm must not find a prompt in the cache because the other arm (or an earlier run)
already sent it, so every conversation gets a salt per run, arm, repeat and seed:
  text   (default) a first line "Session: <salt>" in the system message. Works on every engine; the prompt text
         differs between arms, so T=0 outputs differ even for identical configurations (no OUTPUT IDENTITY, and
         decode tok/s carries trajectory luck like any trajectory-changing lever).
  field  the request field `cache_salt` (vLLM); the text is identical in both arms. The run checks that turn 1
         is not served from the cache (when the engine reports cached tokens) and warns otherwise.
  none   no isolation: only for arms on separate, freshly started servers.
"""
import hashlib
import json
import os

from . import prompts as P

DEFAULT_SET = os.path.join(P.PKG, "..", "prompts", "speedbench-agent-v1.json")
MODES = ("agent", "agent-quick")


def load(path=None):
    path = path or DEFAULT_SET
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def conversations(body, mode, skip_tools=False):
    out = []
    for c in body["conversations"]:
        if mode == "agent-quick" and not c.get("quick"):
            continue
        if skip_tools and c.get("tools"):
            continue
        n = 1 + len(c["turns"])
        if mode == "agent-quick":
            n = min(n, body.get("quick_turns", 4))
        out.append((c, n))
    return out


def request_messages(body, conv, turn, salt_line=None):
    """messages of request `turn` (1-based) of a conversation"""
    system = body["system"] if salt_line is None else salt_line + "\n\n" + body["system"]
    msgs = [{"role": "system", "content": system}] + [dict(m) for m in conv["messages"]]
    for t in conv["turns"][:turn - 1]:
        msgs.append(t["assistant"])
        msgs.extend(t["next"])
    return msgs


def plan(body, a):
    """jobs: repeats x conversations x turns x seeds, in conversation order (turn t right after turn t-1)"""
    seeds = [None] if a.temperature == 0 else [int(s) for s in a.seeds.split(",")]
    jobs = []
    for rep in range(a.repeats):
        for conv, n in conversations(body, a.mode, a.skip_tools):
            for t in range(1, n + 1):
                for s in seeds:
                    pid = "%s@t%d" % (conv["id"], t)
                    jobs.append({"prompt": {"id": pid, "category": "t%d" % t, "conv": conv["id"], "turn": t},
                                 "seed": s, "variant": 0, "repeat": rep, "kind": "measure",
                                 "agent": {"conv": conv, "turn": t}})
    return jobs


def salt(nonce, arm_label, conv_id, repeat, seed):
    return hashlib.sha256(("%s|%s|%s|%s|%s" % (nonce, arm_label, conv_id, repeat, seed)).encode()).hexdigest()[:16]


def materialize(body, job, arm, a):
    """the prompt of an agent job as sent by one arm (salted per --isolate)"""
    conv, t = job["agent"]["conv"], job["agent"]["turn"]
    s = salt(a.agent_nonce, arm.label, conv["id"], job["repeat"], job["seed"])
    p = {"id": job["prompt"]["id"], "category": job["prompt"]["category"]}
    p["messages"] = request_messages(body, conv, t, "Session: %s" % s if a.isolate == "text" else None)
    if a.isolate == "field":
        p["cache_salt"] = s
    if conv.get("tools"):
        p["tools"] = body["tools"]
    return p


def probe_prompts(body, a):
    """turn-1 prompts for the capability/tool probe in the warm-up (salted, never reused)"""
    out = []
    for conv, _ in conversations(body, a.mode, a.skip_tools):
        p = {"id": "probe", "category": "probe",
             "messages": request_messages(body, conv, 1, "Session: probe-%s" % a.agent_nonce)}
        if conv.get("tools"):
            p["tools"] = body["tools"]
        out.append(p)
    return out


CACHE_FLAGS = ("start vLLM or gr8 with --enable-prompt-tokens-details, SGLang with --enable-cache-report, or run "
               "with --metrics where /metrics exports cached prompt tokens")


TURN1_MISS_TOKENS, TURN1_MISS_FRAC = 64, 0.02     # a first turn "misses" if at most this much was served cached


def summarize(recs, isolate=None):
    """per turn index: TTFT, prompt / cached / new tokens, cache hit vs an ideal cache (the previous request's
    prompt), decode tok/s and end-to-end turn latency; and the same over cached turns (t >= 2)"""
    meas = [r for r in recs if r.get("kind") == "measure" and r.get("ok") and r.get("turn")]
    chains = {}
    for r in meas:
        chains.setdefault((r["conv"], r["repeat"], r.get("seed")), {})[r["turn"]] = r
    rows = []
    for ch in chains.values():
        for t, r in ch.items():
            prev = ch.get(t - 1)
            cached = r.get("cached_tokens")
            if cached is None and (r.get("spec") or {}).get("cached_tokens") is not None:
                cached = r["spec"]["cached_tokens"]       # /metrics delta (c1: this request only)
            row = {"turn": t, "conv": r["conv"], "ttft_s": r.get("ttft_s"), "total_s": r.get("total_s"), "prompt": r.get("prompt_tokens"),
                   "cached": cached, "tok_s": r.get("tok_s") if r.get("valid_decode") else None,
                   "ideal": prev.get("prompt_tokens") if prev and t > 1 else 0}
            if row["prompt"] is not None and cached is not None:
                row["new"] = row["prompt"] - cached
            rows.append(row)

    def med(v, nd=3):
        v = sorted(x for x in v if x is not None)
        if not v:
            return None
        m = len(v) // 2
        x = round(v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2, nd)
        return int(x) if nd == 0 else x

    def block(rs):
        out = {"n": len(rs), "median_ttft_ms": med([r["ttft_s"] * 1000 for r in rs if r["ttft_s"] is not None], 1),
               "median_e2e_s": med([r["total_s"] for r in rs]), "median_prompt_tokens": med([r["prompt"] for r in rs], 0),
               "median_cached_tokens": med([r["cached"] for r in rs], 0),
               "median_new_tokens": med([r.get("new") for r in rs], 0),
               "median_tok_s": med([r["tok_s"] for r in rs], 2)}
        hit = [(r["cached"], r["ideal"]) for r in rs if r["cached"] is not None and r["ideal"]]
        if hit:
            out["cache_hit_vs_ideal"] = round(sum(c for c, _ in hit) / sum(i for _, i in hit), 4)
        cr = [(r["cached"], r["prompt"]) for r in rs if r["cached"] is not None and r["prompt"]]
        if cr:
            out["cached_frac_of_prompt"] = round(sum(c for c, _ in cr) / sum(p for _, p in cr), 4)
        nt = [(r["new"], r["ttft_s"]) for r in rs if r.get("new") and r["ttft_s"]]
        if nt:
            out["median_new_tok_per_s_prefill"] = med([n / t for n, t in nt], 1)
        return out
    out = {"per_turn": {}, "reports_cached_tokens": any(r["cached"] is not None for r in rows)}
    for t in sorted({r["turn"] for r in rows}):
        out["per_turn"]["t%d" % t] = block([r for r in rows if r["turn"] == t])
    out["cold"] = block([r for r in rows if r["turn"] == 1])
    out["cached"] = block([r for r in rows if r["turn"] > 1])
    # isolation evidence per conversation: every conversation's first turn must miss the cache (an average would
    # hide one fully contaminated conversation among clean ones)
    t1 = [r for r in rows if r["turn"] == 1]
    known = [r for r in t1 if r["cached"] is not None and r["prompt"]]
    # every conversation of the run, including those whose first turn failed (a failed turn is no evidence)
    convs = sorted({r["conv"] for r in recs if r.get("kind") == "measure" and r.get("conv")})
    if known:
        out["turn1_cached_frac"] = round(sum(r["cached"] for r in known) / sum(r["prompt"] for r in known), 4)
    hit = sorted({r["conv"] for r in known if r["cached"] > max(TURN1_MISS_TOKENS, TURN1_MISS_FRAC * r["prompt"])})
    unknown = sorted(set(convs) - {r["conv"] for r in known})
    if hit:
        out["warning"] = ("turn 1 was served from the prefix cache in %s: the cache isolation did not work there "
                          "(with --isolate field the engine may ignore cache_salt; use --isolate text)" % ", ".join(hit))
        out["isolation"] = "FAILED in %d of %d conversations: %s" % (len(hit), len(convs), ", ".join(hit))
    elif unknown:
        # without cached-token counts nothing shows whether an arm reused the other arm's cache
        out["isolation"] = ("UNVERIFIED for %s: no successful first turn with cached-token counts (a failed first "
                            "turn is no evidence; for missing counts: %s)" % (
                                "all conversations" if not known else ", ".join(unknown), CACHE_FLAGS))
    else:
        out["isolation"] = "verified: turn 1 missed the cache in every conversation (%d)" % len(convs)
    if isolate == "none":
        out["isolation"] = "not isolated (--isolate none); " + out["isolation"]
    return out
