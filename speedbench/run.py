"""speedbench run: measure one or two OpenAI-compatible endpoints over a frozen workload.

Modes: full / quick (the decode prompt set, one request per prompt) and agent / agent-quick (multi-turn agent
conversations that grow turn by turn and exercise the prefix cache; see speedbench/agent.py). Options add engine
counters (--metrics), energy per token (--power-cmd) and a quality guard (--quality-guard)."""
import argparse
import copy
import datetime
import json
import math
import os
import platform
import queue
import socket
import statistics
import sys
import threading
import time
import uuid

from . import __version__, agent, client, engines, guard, longdoc, meta as M, power, prompts as P

MIN_DECODE_TOKENS = 32
# request fields the benchmark controls; --extra-body / arm extras may not override them
PROTECTED = {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens", "temperature",
             "top_p", "seed", "tools", "tool_choice", "reasoning_effort", "n", "ignore_eos", "min_tokens",
             "return_spec_tokens_details", "return_step_stats", "cache_salt"}
# request field that asks an engine for per-request step counters in the response
STATS_FIELD = {"sglang": "return_spec_tokens_details", "gr8": "return_step_stats"}


# --------------------------------------------------------------------------------------------- endpoint
class Arm:
    def __init__(self, label, url, model=None, extra=None, headers=None):
        self.label, self.url, self.model = label, url.rstrip("/"), model
        self.extra = extra or {}
        self.headers = headers or {}
        self.continuous_usage = True
        self.meta = {}
        self.engine = "unknown"
        self.stats_field = None      # STATS_FIELD entry sent with every request, if any


def parse_arm(spec, default_extra, headers):
    """LABEL=URL[,model=NAME][,extra=JSON]  (extra is merged over --extra-body)"""
    label, _, rest = spec.partition("=")
    parts = rest.split(",", 1)
    url, model, extra = parts[0], None, dict(default_extra)
    tail = parts[1] if len(parts) > 1 else ""
    while tail:
        if tail.startswith("model="):
            v, _, tail = tail[6:].partition(",")
            model = v
        elif tail.startswith("extra="):
            extra.update(json.loads(tail[6:]))
            tail = ""
        else:
            raise SystemExit("bad --arm spec: %r" % spec)
    return Arm(label, url, model, extra, headers)


def probe_metadata(arm, timeout=10):
    meta = {"url": arm.url}
    try:
        models, hdr = client.http_json(arm.url + "/v1/models", timeout=timeout, headers=arm.headers)
        meta["models"] = models
        meta["server_header"] = hdr.get("Server") or hdr.get("server")
        if not arm.model:
            arm.model = models["data"][0]["id"]
    except Exception as e:  # noqa: BLE001 - metadata is best effort
        meta["models_error"] = repr(e)[:200]
    for path in ("/version", "/props", "/get_model_info", "/server_info"):
        try:
            txt = client.http_text(arm.url + path, timeout=timeout, headers=arm.headers)
            meta[path] = json.loads(txt) if txt.strip().startswith("{") else txt[:2000]
        except Exception:  # noqa: BLE001
            pass
    if not arm.model:
        raise SystemExit("could not discover a model id at %s/v1/models; pass --model" % arm.url)
    meta["model"] = arm.model
    return meta


def scrape_spec(arm):
    return engines.scrape(arm.url, arm.headers)


def setup_engine(arm, a):
    """detect the engine and decide whether to ask for per-request step counters"""
    counters = scrape_spec(arm) if getattr(a, "metrics", False) else None
    arm.engine = getattr(a, "engine", None) or engines.detect(arm.meta, counters)
    mode = getattr(a, "per_request_stats", "auto")
    if mode == "on" or (mode == "auto" and arm.engine == "sglang" and getattr(a, "metrics", False)):
        arm.stats_field = STATS_FIELD.get(arm.engine)
    arm.meta["engine"] = arm.engine
    arm.meta["metrics_families"] = sorted(counters) if counters else []
    return counters


# --------------------------------------------------------------------------------------------- requests
def build_body(arm, prompt, a, max_tokens, seed=None):
    body = {"model": arm.model, "messages": copy.deepcopy(prompt["messages"]), "stream": True,
            "max_tokens": max_tokens, "temperature": a.temperature,
            "stream_options": {"include_usage": True}}
    if arm.continuous_usage:
        body["stream_options"]["continuous_usage_stats"] = True
    if a.top_p is not None:
        body["top_p"] = a.top_p
    if seed is not None:
        body["seed"] = seed
    if prompt.get("tools"):
        body["tools"] = prompt["tools"]
    if prompt.get("cache_salt"):
        body["cache_salt"] = prompt["cache_salt"]          # agent mode, --isolate field (vLLM)
    if a.effort and a.effort_field in ("template", "both"):
        body["chat_template_kwargs"] = {"reasoning_effort": a.effort}
    if a.effort and a.effort_field in ("top", "both"):
        body["reasoning_effort"] = a.effort
    if arm.stats_field:
        body[arm.stats_field] = True
    for k, v in arm.extra.items():
        if isinstance(v, dict) and isinstance(body.get(k), dict):
            body[k] = dict(body[k], **v)
        else:
            body[k] = v
    body["max_tokens"] = max_tokens          # never overridable (the priming request must stay at 1 token)
    return body


def check_extra(extra, where):
    bad = sorted(set(extra) & PROTECTED)
    ctk = extra.get("chat_template_kwargs") or {}
    if "reasoning_effort" in ctk:
        bad.append("chat_template_kwargs.reasoning_effort")
    if bad:
        raise SystemExit("%s may not set benchmark-controlled fields %s; use the dedicated options" % (where, bad))


def one(arm, prompt, a, max_tokens, seed=None, metrics=False):
    body = build_body(arm, prompt, a, max_tokens, seed)
    before = scrape_spec(arm) if metrics else None
    try:
        r = client.stream_chat(arm.url, body, timeout=a.timeout, headers=arm.headers)
    except client.RequestFailed as e:
        return {"ok": False, "error": str(e)[:500]}
    except Exception as e:  # noqa: BLE001 - one bad request must never lose the run
        return {"ok": False, "error": "client error: %r" % (e,)}
    after = scrape_spec(arm) if metrics else None
    r["ok"] = True
    d = engines.delta(before, after)
    if d:
        r["spec"] = d
    pr = engines.per_request(r.pop("extras", None))
    if pr:
        r["spec_request"] = pr
    return r


def derive(r):
    """Per-request headline numbers. k1 = tokens in the first text chunk (1 unless the server reports it)."""
    if not r.get("ok"):
        return r
    n = r["completion_tokens"]
    if r.get("boundary") == "exact":
        dt = r["last_chunk_tokens"] - r["first_chunk_tokens"]     # tokens generated inside the timed interval
    else:
        dt = n - 1                                                # the documented fallback when counts are unknown
    steps = max(0, r["n_steps"] if r.get("n_steps") is not None else r["n_chunks"] - 1)
    r["decode_tokens"] = dt
    r["tok_s"] = dt / r["decode_s"] if r["decode_s"] and r["decode_s"] > 0 and dt > 0 else None
    r["steps_stream"] = steps
    decompose(r, steps, dt)
    r["valid_decode"] = r["tok_s"] is not None and r["decode_tokens"] >= MIN_DECODE_TOKENS
    return r


def decompose(r, steps_stream, dt):
    """tok/s = tokens per step x steps per second, exactly, per request.

    Engine steps (per-request response fields first, then the /metrics delta around the request) cover the whole
    request, so tokens/step = (completion_tokens - 1) / steps (the first token comes from the prefill) and the
    step time is the client decode interval over the steps that fall inside it, steps x dt / (completion - 1).
    Their ratio is exactly the decode tok/s. Without engine counters, stream events inside the interval are the
    steps (valid only where the server sends one event per step)."""
    spec = r.get("spec_request") or r.get("spec") or {}
    src = "response" if r.get("spec_request") else "metrics"
    if src == "metrics" and spec and spec.get("finished") not in (None, 1):
        # the /metrics delta saw other requests finish (another client), or not yet this one: not attributable
        r["spec_note"] = "metrics delta not attributable: %s requests finished during it" % spec.get("finished")
        spec = {}
    n = (r.get("completion_tokens") or 0) - 1
    st = engines.steps_of(spec)
    if r.get("spec") and r["spec"].get("spec_steps"):
        r["steps_metrics"] = r["spec"]["spec_steps"]
    if st and n > 0 and dt > 0 and r.get("decode_s"):
        r["steps_engine"] = st
        r["steps_source"] = src
        inside = st * dt / n
        r["step_ms"] = r["decode_s"] * 1000 / inside
        r["tokens_per_step"] = n / st
        if spec.get("draft_tokens"):
            r["draft_per_step"] = spec["draft_tokens"] / st
            if spec.get("accepted") is not None:
                r["accept_rate"] = spec["accepted"] / spec["draft_tokens"]
        # server-side step time: the engine's own decode time of this request (first token -> end) per step;
        # free of client/network jitter. Only when the delta covers exactly one finished request.
        if spec.get("decode_s") and (src == "response" or spec.get("finished") == 1):
            r["step_ms_server"] = spec["decode_s"] * 1000 / st
    elif steps_stream > 0 and r.get("decode_s"):
        r["steps_source"] = "stream"
        r["step_ms"] = r["decode_s"] * 1000 / steps_stream
        r["tokens_per_step"] = dt / steps_stream


def prepare(arm, a, prompts=(), long_tokens=None, cpt=3.6):
    """capability probe for per-chunk usage, tool probe, warm-up and one long cold prefill (all discarded)"""
    if arm.continuous_usage:
        pr = one(arm, {"messages": P.WARMUP[0]}, a, 8)
        if not pr.get("ok"):
            arm.continuous_usage = False
            pr = one(arm, {"messages": P.WARMUP[0]}, a, 8)
            if not pr.get("ok"):
                raise SystemExit("%s: probe request failed: %s" % (arm.label, pr.get("error")))
    tool_p = next((p for p in prompts if p.get("tools")), None)
    if tool_p is not None:
        pr = one(arm, tool_p, a, 16)
        if not pr.get("ok"):
            raise SystemExit("%s: the server rejected a prompt with `tools` and tool-call history (%s); "
                             "configure its tool parser or run with --skip-tools" % (arm.label, pr.get("error")))
    for i in range(a.warmup):
        w = one(arm, {"messages": P.WARMUP[i % len(P.WARMUP)]}, a, 64)
        if not a.quiet:
            print("%-4s warm-up %d: %s" % (arm.label, i + 1, "ok" if w.get("ok") else w.get("error")),
                  file=sys.stderr, flush=True)
    long_warmup(arm, a, a.warmup_prefill if long_tokens is None else long_tokens, cpt=cpt)


def long_warmup(arm, a, tokens, log=True, cpt=3.6):
    """one long cold prefill (unique document, max_tokens=1): the first long prefill after a boot runs on cold
    kernels and graphs and must never be a timed one"""
    if not tokens:
        return None
    doc, _, _ = longdoc.sized(arm.url, arm.model, tokens, cpt, seed=4242, salt="warm-up %s" % uuid.uuid4().hex,
                              headers=arm.headers, tol=0.05)
    w = one(arm, {"messages": longdoc.messages(doc, longdoc.QUESTIONS[0])}, a, 1)
    if log and not getattr(a, "quiet", False):
        print("%-4s warm-up long prefill: %s" % (arm.label, "ok, %s prompt tokens, %.1f s" % (
            w.get("prompt_tokens"), w.get("ttft_s") or 0) if w.get("ok") else w.get("error")), file=sys.stderr, flush=True)
    return w


# --------------------------------------------------------------------------------------------- schedule
def with_variant(p, v):
    """T=0 trajectory variant v >= 1: a short tag appended to the last user message gives every prompt a
    different, still deterministic trajectory (more luck samples without new prompts). The prefix (system
    message, documents, tool history) is unchanged, so the prefix cache stays shared across variants."""
    if not v:
        return p
    q = copy.deepcopy(p)
    for m in reversed(q["messages"]):
        if m["role"] == "user":
            m["content"] = m["content"] + "\n\n(Request tag: %d)" % v
            break
    return q


def plan(prompts, a):
    seeds = [None] if a.temperature == 0 else [int(s) for s in a.seeds.split(",")]
    jobs = []
    for rep in range(a.repeats):
        for p in prompts:
            for v in range(a.variants):
                for s in seeds:
                    jobs.append({"prompt": with_variant(p, v), "seed": s, "variant": v, "repeat": rep,
                                 "kind": "measure"})
    sentinels = [dict(j, kind="sentinel") for j in jobs[:a.sentinels]] if a.conc == 1 else []
    return jobs, sentinels


def key_of(job):
    k = job["prompt"]["id"]
    if job.get("variant"):
        k += "@v%d" % job["variant"]
    if job["seed"] is not None:
        k += "@s%d" % job["seed"]
    return k


def run_jobs(arms, jobs, a, log, results):
    """c1: strictly sequential; with two arms the order alternates A B / B A per job (ABBA blocks)."""
    for n, job in enumerate(jobs):
        order = arms if (n % 2 == 0 or len(arms) == 1) else arms[::-1]
        for arm in order:
            p = job["prompt"]
            rec = {"key": key_of(job), "id": p["id"], "category": p["category"], "seed": job["seed"],
                   "variant": job.get("variant", 0), "repeat": job["repeat"], "kind": job["kind"], "pos": n,
                   "t_wall": round(time.time() - a.t0, 2)}
            if job.get("agent"):
                p = agent.materialize(a.agent_body, job, arm, a)
                rec["conv"], rec["turn"] = job["agent"]["conv"]["id"], job["agent"]["turn"]
            if a.prime and a.conc == 1:
                pr = one(arm, p, a, 1, job["seed"])
                rec["prime"] = {k: pr.get(k) for k in ("ok", "ttft_s", "prompt_tokens", "cached_tokens", "error")}
            r = derive(one(arm, p, a, a.max_tokens, job["seed"], metrics=a.metrics))
            t_send = r.pop("t_send", None)
            if getattr(a, "sampler", None) is not None and t_send is not None:
                rec["t_send_rel"] = round(t_send - a.sampler.origin, 4)
            if not a.keep_text:
                r.pop("text", None)
            if not a.keep_chunks:
                r.pop("chunks", None)
            rec.update(r)
            results[arm.label].append(rec)
            log(arm, rec, n, len(jobs))


def run_concurrent(arm, jobs, a, log, results):
    """c>1: c workers pull jobs from one queue in the fixed order (per-request numbers stay paired by key)."""
    q = queue.Queue()
    for n, j in enumerate(jobs):
        q.put((n, j))
    lock = threading.Lock()
    wall = {"start": time.perf_counter()}

    def worker():
        while True:
            try:
                n, job = q.get_nowait()
            except queue.Empty:
                return
            p = job["prompt"]
            rec = {"key": key_of(job), "id": p["id"], "category": p["category"], "seed": job["seed"],
                   "variant": job.get("variant", 0), "repeat": job["repeat"], "kind": job["kind"], "pos": n}
            r = derive(one(arm, p, a, a.max_tokens, job["seed"]))
            r.pop("text", None) if not a.keep_text else None
            r.pop("chunks", None) if not a.keep_chunks else None
            t_send = r.pop("t_send", None)
            if t_send is not None:
                rec["t_start_s"] = t_send - wall["start"]
            rec.update(r)
            with lock:
                results[arm.label].append(rec)
                log(arm, rec, n, len(jobs))

    ts = [threading.Thread(target=worker) for _ in range(a.conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return time.perf_counter() - wall["start"]


# --------------------------------------------------------------------------------------------- summary
def summarize(recs, conc_wall=None, agent_mode=False, idle_w=None, isolate=None):
    meas = [r for r in recs if r["kind"] == "measure"]
    ok = [r for r in meas if r.get("ok")]
    out = {"requests": len(meas), "failed": len(meas) - len(ok),
           "valid_decode": sum(1 for r in ok if r.get("valid_decode")),
           "exact_boundary": sum(1 for r in ok if r.get("boundary") == "exact"),
           "errors": sorted({r.get("error", "")[:120] for r in meas if not r.get("ok")})[:5]}
    cats = sorted({r["category"] for r in meas})
    for c in cats + ["ALL"]:
        rs = [r for r in ok if (c == "ALL" or r["category"] == c) and r.get("valid_decode")]
        if not rs:
            continue
        toks = [r["tok_s"] for r in rs]
        s = {"n": len(rs), "median_tok_s": round(statistics.median(toks), 2),
             "geomean_tok_s": round(math.exp(statistics.fmean(math.log(x) for x in toks)), 2),
             "aggregate_tok_s": round(sum(r["decode_tokens"] for r in rs) / sum(r["decode_s"] for r in rs), 2),
             "median_ttft_ms": round(statistics.median(r["ttft_s"] for r in rs) * 1000, 1),
             "truncated_frac": round(sum(r.get("finish_reason") == "length" for r in rs) / len(rs), 3)}
        rt = [r["reasoning_tokens"] / r["completion_tokens"] for r in rs if r.get("reasoning_tokens") is not None]
        if rt:
            s["reasoning_token_frac"] = round(statistics.fmean(rt), 3)
        rc = [r["reasoning_chars"] / max(1, r["reasoning_chars"] + r["content_chars"]) for r in rs]
        s["reasoning_char_frac"] = round(statistics.fmean(rc), 3)
        st = [r["step_ms"] for r in rs if r.get("step_ms")]
        if st:
            s["median_step_ms"] = round(statistics.median(st), 2)
            s["median_tokens_per_step"] = round(statistics.median(r["tokens_per_step"] for r in rs
                                                                  if r.get("tokens_per_step")), 3)
            s["steps_source"] = "/".join(sorted({r.get("steps_source") for r in rs if r.get("steps_source")}))
        for f, nd in (("accept_rate", 4), ("draft_per_step", 3), ("step_ms_server", 3)):
            v = [r[f] for r in rs if r.get(f) is not None]
            if v:
                s["median_" + f] = round(statistics.median(v), nd)
        pr = [r["prime"]["ttft_s"] for r in rs if r.get("prime") and r["prime"].get("ttft_s")]
        if pr:
            s["median_prime_ttft_ms"] = round(statistics.median(pr) * 1000, 1)
        en = power.summarize(rs, idle_w)
        if en:
            s["energy"] = en
        out[c] = s
    if agent_mode:
        out["agent"] = agent.summarize(recs, isolate)
    if conc_wall:
        out["aggregate_throughput_tok_s"] = round(sum(r["completion_tokens"] for r in ok) / conc_wall, 2)
        out["wall_s"] = round(conc_wall, 1)
    sent = [r for r in recs if r["kind"] == "sentinel" and r.get("ok")]
    if sent:
        orig = {r["key"]: r for r in ok if r["repeat"] == 0}
        pairs = [(orig[s["key"]], s) for s in sent if s["key"] in orig and s.get("tok_s") and orig[s["key"]].get("tok_s")]
        if pairs:
            out["sentinel"] = {
                "n": len(pairs),
                "identical_text": sum(o["sha256"] == s["sha256"] for o, s in pairs),
                "drift_pct_identical": _drift([(o, s) for o, s in pairs if o["sha256"] == s["sha256"]]),
                "drift_pct_all": _drift(pairs)}
    reps = {}
    for r in ok:
        reps.setdefault(r["key"], set()).add(r["sha256"])
    multi = [k for k in reps if sum(1 for r in ok if r["key"] == k) > 1]
    if multi:
        out["repeat_identical_frac"] = round(sum(len(reps[k]) == 1 for k in multi) / len(multi), 3)
    return out


def _drift(pairs):
    if not pairs:
        return None
    return round((math.exp(statistics.fmean(math.log(s["tok_s"] / o["tok_s"]) for o, s in pairs)) - 1) * 100, 3)


# --------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="speedbench run", description=__doc__)
    ap.add_argument("--url", help="endpoint base URL (single-arm mode), e.g. http://127.0.0.1:8000")
    ap.add_argument("--model", help="model id (default: first id from /v1/models)")
    ap.add_argument("--arm", action="append", default=[],
                    help="two-arm interleaved mode: LABEL=URL[,model=NAME][,extra=JSON]; give it twice")
    ap.add_argument("--label", default=None, help="label of this run (single-arm mode)")
    ap.add_argument("--block", default=None, help="boot/block id recorded in the result (e.g. boot1)")
    ap.add_argument("--out", default=None, help="output JSON (single arm) or prefix (two arms)")
    ap.add_argument("--mode", choices=["full", "quick", "agent", "agent-quick"], default="full",
                    help="full: 73 prompts x 256 tokens; quick: 30 x 160 (screen); agent: 8 multi-turn conversations, "
                         "48 requests x 160 tokens (prefix cache); agent-quick: 4 conversations x 4 turns")
    ap.add_argument("--isolate", choices=["text", "field", "none"], default="text",
                    help="agent modes: prefix-cache isolation between arms/runs: a salt line in the system message "
                         "(text), the cache_salt request field (field, vLLM), or none (separate fresh servers)")
    ap.add_argument("--prompts", default=None, help="prompt-set JSON (default: the shipped speedbench-v1)")
    ap.add_argument("--categories", default=None, help="comma list to restrict categories")
    ap.add_argument("--skip-tools", action="store_true", help="drop prompts that send a `tools` array")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 256 (full) / 160 (quick)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seeds", default="1,2,3", help="seeds used when --temperature > 0")
    ap.add_argument("--effort", default="low", help="reasoning effort; '' to send none")
    ap.add_argument("--effort-field", choices=["template", "top", "both"], default="template",
                    help="template: chat_template_kwargs.reasoning_effort (GLM, gr8); top: OpenAI reasoning_effort")
    ap.add_argument("--extra-body", default="{}", help="JSON merged into every request body")
    ap.add_argument("--header", action="append", default=[], help="extra HTTP header 'Name: value'")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding a bearer token, if set")
    ap.add_argument("--conc", type=int, default=1, help="concurrent streams (1 = the A/B measurement)")
    ap.add_argument("--repeats", type=int, default=1, help="send every job N times (adds trajectories only if "
                    "the engine is nondeterministic at T=0)")
    ap.add_argument("--variants", type=int, default=1,
                    help="T=0 trajectory variants per prompt (tag on the last user message); 4 can at best halve "
                         "the luck part of the CI, at 4x the time")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sentinels", type=int, default=None, help="re-run the first N jobs at the end (default 4/2)")
    ap.add_argument("--no-prime", dest="prime", action="store_false",
                    help="do not send the max_tokens=1 prefix-cache priming request before each measured one")
    ap.add_argument("--metrics", action="store_true",
                    help="scrape /metrics engine counters around each request (speed decomposition: steps, "
                         "tokens per step, acceptance, server-side step time)")
    ap.add_argument("--per-request-stats", choices=["auto", "on", "off"], default="auto",
                    help="ask for per-request step counters in the response (auto: SGLang with --metrics; on: also "
                         "gr8's proposed return_step_stats)")
    ap.add_argument("--preregister", choices=["tok", "step"], default="tok",
                    help="decision rule declared BEFORE the run: tok = the tok/s verdict decides; step = this is a "
                         "step-cost lever (kernels, communication, scheduling) and the step-time verdict decides when "
                         "the acceptance checks pass (compare: STEP-TIME RULE)")
    ap.add_argument("--tps-margin", type=float, default=2.0,
                    help="pre-registered equivalence margin in %% for tokens/step and drafts/step (step-time rule)")
    ap.add_argument("--warmup-prefill", type=int, default=8192,
                    help="tokens of one long cold prefill in the warm-up, before anything is timed (0 = none)")
    ap.add_argument("--no-continuous-usage", action="store_true",
                    help="do not ask for per-chunk usage (stream_options.continuous_usage_stats)")
    ap.add_argument("--keep-text", action="store_true", help="store generated text (hashes are always kept)")
    ap.add_argument("--no-chunks", dest="keep_chunks", action="store_false", help="do not store chunk traces")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--quiet", action="store_true")
    g = ap.add_argument_group("quality guard (optional)")
    g.add_argument("--quality-guard", action="store_true",
                   help="after the timed workload, run the 30 exact-answer guard items at T=0 in every arm, and keep "
                        "the generated text for the divergence summary in compare (implies --keep-text)")
    g.add_argument("--guard-max-tokens", type=int, default=512, help="max_tokens of a guard item (thinking included)")
    g.add_argument("--guard-set", default=None, help="guard item file (default: the shipped speedbench-guard-v1)")
    power.add_args(ap)
    M.add_args(ap)
    a = ap.parse_args(argv)
    agent_mode = a.mode in agent.MODES
    if a.max_tokens is None:
        a.max_tokens = 256 if a.mode == "full" else 160
    if a.sentinels is None:
        a.sentinels = 4 if a.mode == "full" else 0 if agent_mode else 2
    if agent_mode:
        if a.variants != 1:
            raise SystemExit("--variants is not available in the agent modes (use --temperature with --seeds)")
        if a.conc != 1:
            raise SystemExit("the agent modes are c1 (one conversation at a time)")
        a.prime = False          # a priming request would put the new suffix in the cache before it is timed
        a.sentinels = 0          # a re-sent turn would be served from the cache
    if a.quality_guard:
        a.keep_text = True
    headers = {}
    for h in a.header:
        k, _, v = h.partition(":")
        headers[k.strip()] = v.strip()
    if os.environ.get(a.api_key_env):
        headers.setdefault("Authorization", "Bearer " + os.environ[a.api_key_env])
    extra = json.loads(a.extra_body)
    check_extra(extra, "--extra-body")
    if a.arm:
        if a.url or len(a.arm) != 2:
            raise SystemExit("two-arm mode needs exactly two --arm and no --url")
        arms = [parse_arm(s, extra, headers) for s in a.arm]
        for arm in arms:
            check_extra(arm.extra, "--arm %s extra" % arm.label)
        if arms[0].label == arms[1].label:
            raise SystemExit("the two --arm labels must differ")
        if a.conc != 1:
            raise SystemExit("two-arm interleaved mode is c1 only")
    elif a.url:
        label = a.label or (os.path.splitext(os.path.basename(a.out))[0] if a.out else "run")
        arms = [Arm(label, a.url, a.model, extra, headers)]
    else:
        raise SystemExit("give --url or two --arm")
    for arm in arms:
        arm.continuous_usage = not a.no_continuous_usage

    cats = a.categories.split(",") if a.categories else None
    a.agent_nonce = uuid.uuid4().hex[:12]
    if agent_mode:
        body, set_sha = agent.load(a.prompts)
        a.agent_body = body
        jobs, sentinels = agent.plan(body, a), []
        if cats:
            jobs = [j for j in jobs if j["prompt"]["category"] in cats]
        prompts = agent.probe_prompts(body, a)
        n_selected = len({j["prompt"]["id"] for j in jobs})
    else:
        body, set_sha = P.load(a.prompts)
        prompts = P.select(body, a.mode, a.skip_tools, cats)
        jobs, sentinels = plan(prompts, a)
        n_selected = len(prompts)
    if not jobs:
        raise SystemExit("nothing selected")
    t0 = a.t0 = time.time()
    counters = {}
    for arm in arms:
        arm.meta = probe_metadata(arm)
        counters[arm.label] = setup_engine(arm, a)

    def log(arm, rec, n, total):
        if a.quiet:
            return
        if rec.get("ok"):
            msg = "%-4s %3d/%d %-16s %6s tok/s  ttft %5.0f ms  %4d tok%s" % (
                arm.label, n + 1, total, rec["key"], "%.1f" % rec["tok_s"] if rec.get("tok_s") else "-",
                (rec["ttft_s"] or 0) * 1000, rec["completion_tokens"], "" if rec["kind"] == "measure" else " (sentinel)")
        else:
            msg = "%-4s %3d/%d %-16s FAILED %s" % (arm.label, n + 1, total, rec["key"], rec.get("error"))
        print(msg, file=sys.stderr, flush=True)

    a.sampler = power.start(a)
    try:
        for arm in arms:
            prepare(arm, a, prompts)
        idle_w = power.idle(a.sampler, a.power_idle)
        results = {arm.label: [] for arm in arms}
        wall = None
        t_phase = time.perf_counter()
        if a.conc == 1:
            run_jobs(arms, jobs + sentinels, a, log, results)
        else:
            wall = run_concurrent(arms[0], jobs, a, log, results)
        t_phase_end = time.perf_counter()
        guards = None
        if a.quality_guard:
            def glog(arm, rec):
                if not a.quiet:
                    print("%-4s guard %-10s %s" % (arm.label, rec["id"], "pass" if rec["pass"] else "FAIL %r" % (
                        rec.get("answer") or rec.get("error") or rec.get("note"),)), file=sys.stderr, flush=True)
            guards = guard.run_items(arms, a, one, glog)
    finally:
        if a.sampler is not None:
            time.sleep(1.0)      # let the last readings arrive before the sources stop
            a.sampler.stop()
    energy = None
    if a.sampler is not None:
        energy = a.sampler.describe(keep_samples=not a.power_no_samples)
        energy.update({"note": power.NOTE[a.power_kind], "idle_w": idle_w, "per_request": a.conc == 1})
        e_phase = a.sampler.energy(t_phase, t_phase_end)
        if e_phase is not None:
            energy["phase"] = {"seconds": round(t_phase_end - t_phase, 2), "energy_j": round(e_phase, 1),
                               "mean_w": round(e_phase / max(1e-9, t_phase_end - t_phase), 2)}
        if a.conc == 1:
            for lab in results:
                power.annotate(results[lab], a.sampler)
    run_id = uuid.uuid4().hex
    settings = {k: getattr(a, k) for k in ("mode", "max_tokens", "temperature", "top_p", "effort", "effort_field",
                                           "conc", "repeats", "variants", "prime", "skip_tools",
                                           "categories", "sentinels")}
    settings["seeds"] = a.seeds if a.temperature > 0 else None
    settings["prime"] = bool(a.prime and a.conc == 1)
    settings["extra_body"] = extra
    settings["expected_jobs"] = len(jobs)
    settings["metrics"] = a.metrics
    settings["preregister"] = a.preregister
    settings["tps_margin"] = a.tps_margin
    settings["warmup_prefill"] = a.warmup_prefill
    settings["isolate"] = a.isolate if agent_mode else None
    if a.power_cmd:
        settings["power"] = {"cmds": a.power_cmd, "kind": a.power_kind}
    if a.quality_guard:
        settings["quality_guard"] = {"max_tokens": a.guard_max_tokens}
    paths = []
    for arm in arms:
        recs = sorted(results[arm.label], key=lambda r: r["pos"])
        doc = {
            "schema": "speedbench-result/1", "tool_version": __version__, "label": arm.label, "block": a.block,
            "run_id": run_id + "-" + arm.label, "session_id": run_id, "arms_in_session": [x.label for x in arms],
            "started": datetime.datetime.fromtimestamp(t0).astimezone().isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - t0, 1),
            "prompt_set": {"version": body.get("version"), "sha256": set_sha, "n_selected": n_selected},
            "settings": settings, "arm": {"url": arm.url, "model": arm.model, "extra_body": arm.extra,
                                          "continuous_usage": arm.continuous_usage, "engine": arm.engine,
                                          "stats_field": arm.stats_field},
            "meta": M.collect(a, arm.meta, counters.get(arm.label), mode=a.mode),
            # the effective request minus the arm's own extra fields (those are the treatment, stored in arm)
            "request_template": {k: v for k, v in build_body(arm, prompts[0], a, a.max_tokens).items()
                                 if k not in ("messages", "tools", "cache_salt") and k not in arm.extra},
            "server": arm.meta,
            "client": {"host": socket.gethostname(), "python": platform.python_version(),
                       "platform": platform.platform()},
            "summary": summarize(recs, wall, agent_mode, energy and energy.get("idle_w"), a.isolate), "requests": recs}
        if energy is not None:
            doc["energy"] = energy
            phase = energy.get("phase")
            if phase and len(arms) == 1:
                tok = sum(r.get("completion_tokens") or 0 for r in recs if r.get("ok") and r["kind"] == "measure")
                if tok:
                    doc["summary"]["energy_phase_j_per_tok"] = round(phase["energy_j"] / tok, 5)
        if guards:
            doc["guard"] = guards[arm.label]
        if len(arms) == 1:
            path = a.out or "speedbench-%s-%s.json" % (arm.label, time.strftime("%Y%m%d-%H%M%S"))
        else:
            path = "%s%s.json" % (a.out or "speedbench-", arm.label)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        paths.append(path)
        if a.registry:
            from . import registry
            registry.append(a.registry, doc, path)
        s = doc["summary"]
        print("%s: %s" % (arm.label, json.dumps({k: s[k] for k in s if k in ("ALL", "failed", "valid_decode",
                                                                               "sentinel", "aggregate_throughput_tok_s")})))
        if s.get("agent"):
            ag = s["agent"]
            print("%s: agent: cold TTFT %s ms, cached-turn TTFT %s ms, cache hit vs ideal %s, e2e %s s; cache isolation "
                  "%s" % (arm.label, ag["cold"].get("median_ttft_ms"), ag["cached"].get("median_ttft_ms"),
                          ag["cached"].get("cache_hit_vs_ideal", "n/a"), ag["cached"].get("median_e2e_s"),
                          ag.get("isolation")))
        if doc.get("guard"):
            print("%s: quality guard %d/%d passed" % (arm.label, doc["guard"]["passed"], doc["guard"]["total"]))
    print("wrote %s (%.1f min)" % (", ".join(paths), (time.time() - t0) / 60))
    return paths


if __name__ == "__main__":
    main()
