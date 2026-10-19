"""speedbench load: decode speed under concurrency, and an agentic mix (one long-context session + short arrivals).

Concurrency levels (default c = 1, 2, 4, 8): c closed-loop streams, each sending distinct prompts from the
prompt set (no tools, no long-context prompts) back to back for --duration seconds. Reported per level:
aggregate decode tok/s (generated tokens arriving inside the steady window, from the time every stream had its
first token to the stop time, with a CI over 10-s bins), per-stream decode tok/s (per request, geometric mean
with a CI over requests), TTFT (queueing + prefill), and the engine decomposition of the block from /metrics
before and after it: tokens per step of a stream, server-side step time of a stream, and where the engine
exposes it, engine steps and batch occupancy.

Agentic mix (--mix): one long session (a generated document of --long-tokens, first turn cold, later turns the
same document with a new question, so the prefix cache serves it) decodes --long-max-tokens per turn, while
short requests arrive as a Poisson process (mean gap --short-every s). Reported per class: TTFT and decode tok/s
of both, and the long session's decode tok/s split by what the short requests were doing at that moment
(nothing / waiting or prefilling / decoding).

All numbers are descriptive (one session; streams share the engine, so requests are not independent).
"""
import argparse
import datetime
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import uuid

from . import __version__, engines, longdoc, meta as M, power, prompts as P, run as R
from . import stats as S


def common_args(ap, out_default):
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--block", default=None, help="session/boot label")
    ap.add_argument("--out", default=None, help="output JSON (default %s-LABEL-TIME.json)" % out_default)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--effort-field", choices=["template", "top", "both"], default="template")
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--header", action="append", default=[])
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--metrics", action="store_true", help="scrape /metrics around blocks (decomposition)")
    ap.add_argument("--per-request-stats", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--no-continuous-usage", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--quiet", action="store_true")
    power.add_args(ap)
    M.add_args(ap)


def level_energy(sampler, res, ok_recs):
    """energy of a concurrency level: block (all completion tokens over the block) and the steady window"""
    t_begin, t_end, lo, hi = res.pop("_window")
    if sampler is None:
        return
    e = sampler.energy(t_begin, t_end)
    tok = sum(r.get("completion_tokens") or 0 for r in ok_recs)
    if e is None or not tok:
        return
    out = {"energy_j": round(e, 1), "mean_w": round(e / (t_end - t_begin), 2), "block_j_per_tok": round(e / tok, 5),
           "block_tok_per_j": round(tok / e, 4)}
    if res.get("aggregate_tok_s") and hi - lo > 1:
        ew = sampler.energy(lo, hi)
        if ew:
            out["window_j_per_tok"] = round(ew / (res["aggregate_tok_s"] * (hi - lo)), 5)
    res["energy"] = out


def make_arm(a, kind):
    a.temperature, a.top_p = 0.0, None
    headers = {}
    for h in a.header:
        k, _, v = h.partition(":")
        headers[k.strip()] = v.strip()
    if os.environ.get(a.api_key_env):
        headers.setdefault("Authorization", "Bearer " + os.environ[a.api_key_env])
    extra = json.loads(a.extra_body)
    R.check_extra(extra, "--extra-body")
    label = a.label or (os.path.splitext(os.path.basename(a.out))[0] if a.out else kind)
    arm = R.Arm(label, a.url, a.model, extra, headers)
    arm.continuous_usage = not a.no_continuous_usage
    arm.meta = R.probe_metadata(arm)
    counters = R.setup_engine(arm, a)
    return arm, counters


def log(a, msg):
    if not a.quiet:
        print(msg, file=sys.stderr, flush=True)


def gmean_ci(v):
    """geometric mean and t-CI over a list of positive values (descriptive)"""
    v = [x for x in v if x and x > 0]
    if not v:
        return None
    lg = [math.log(x) for x in v]
    m = statistics.fmean(lg)
    if len(v) < 2:
        return {"n": 1, "gmean": round(v[0], 3), "ci": None}
    se = statistics.stdev(lg) / math.sqrt(len(v))
    t = S.t_ppf(0.975, len(v) - 1)
    return {"n": len(v), "gmean": round(math.exp(m), 3), "ci": [round(math.exp(m - t * se), 3),
                                                                 round(math.exp(m + t * se), 3)]}


def mean_ci(v):
    if not v:
        return None
    if len(v) < 2:
        return {"n": 1, "mean": round(v[0], 3), "ci": None}
    m, se = statistics.fmean(v), statistics.stdev(v) / math.sqrt(len(v))
    t = S.t_ppf(0.975, len(v) - 1)
    return {"n": len(v), "mean": round(m, 3), "ci": [round(m - t * se, 3), round(m + t * se, 3)]}


def pctl(v, q):
    return round(S.quantile(v, q), 4) if v else None


def exact(recs):
    """every chunk of every request carries the cumulative token count (per-chunk usage)"""
    return all(r.get("chunks") and all(c[2] >= 0 for c in r["chunks"]) for r in recs)


def chunk_times(r):
    """[(absolute client time, cumulative tokens)] of a request's text chunks (needs per-chunk usage)"""
    return [(r["t_send"] + t_ms / 1000, cum) for t_ms, _, cum, _ in r.get("chunks") or [] if cum >= 0]


def tokens_in(recs, lo, hi):
    tot = 0.0
    for r in recs:
        prev = 0
        for t, c in chunk_times(r):
            if lo <= t < hi:
                tot += c - prev
            prev = c
    return tot


def block_decomposition(d, wall_s):
    """engine decomposition of a block from the /metrics delta (all requests of the block finished)"""
    if not d:
        return None
    out = {"counters": {k: round(v, 4) for k, v in d.items()}}
    st = d.get("spec_steps")
    if st:
        out["stream_steps"] = st
        if d.get("accepted") is not None:
            out["tokens_per_stream_step"] = round(1 + d["accepted"] / st, 4)
        if d.get("draft_tokens"):
            out["drafts_per_step"] = round(d["draft_tokens"] / st, 4)
            if d.get("accepted") is not None:
                out["accept_rate"] = round(d["accepted"] / d["draft_tokens"], 4)
        if d.get("decode_s"):
            out["stream_step_ms_server"] = round(d["decode_s"] * 1000 / st, 3)
    if d.get("engine_steps"):
        out["engine_steps"] = d["engine_steps"]
        if d.get("engine_step_s"):
            out["engine_step_ms"] = round(d["engine_step_s"] * 1000 / d["engine_steps"], 3)
        if d.get("engine_step_sessions"):
            out["batch_occupancy"] = round(d["engine_step_sessions"] / d["engine_steps"], 3)
    if d.get("iterations"):
        # vLLM's iteration_tokens histogram counts iterations that produced outputs (an outputless chunked-prefill
        # iteration is not counted): a lower bound on engine steps, descriptive only
        out["output_iterations"] = d["iterations"]
        out["wall_ms_per_output_iteration"] = round(wall_s * 1000 / d["iterations"], 3)
        if d.get("iteration_tokens") is not None:
            out["tokens_per_output_iteration"] = round(d["iteration_tokens"] / d["iterations"], 3)
    return out


def short_prompts(body):
    return [p for p in body["prompts"] if not p.get("tools") and p["category"] != "longctx"]


# ------------------------------------------------------------------------------------------- concurrency
def level(arm, a, c, prompts, offset):
    before = engines.scrape(arm.url, arm.headers) if a.metrics else None
    recs, lock = [], threading.Lock()
    nxt = [offset]
    first_tok = {}
    t_begin = time.perf_counter()
    t_stop = t_begin + a.duration

    def worker(w):
        while time.perf_counter() < t_stop:
            with lock:
                p = prompts[nxt[0] % len(prompts)]
                nxt[0] += 1
            r = R.derive(R.one(arm, p, a, a.max_tokens))
            r.update({"id": p["id"], "category": p["category"], "stream": w, "conc": c})
            if r.get("ok") and r.get("ttft_s") is not None:
                with lock:
                    first_tok.setdefault(w, r["t_send"] + r["ttft_s"])
            with lock:
                recs.append(r)
    ts = [threading.Thread(target=worker, args=(w,)) for w in range(c)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    t_end = time.perf_counter()
    after = engines.scrape(arm.url, arm.headers) if a.metrics else None
    ok = [r for r in recs if r.get("ok")]
    lo = max(first_tok.values()) if len(first_tok) == c else t_begin
    hi = t_stop
    res = {"conc": c, "requests": len(recs), "failed": len(recs) - len(ok), "duration_s": a.duration,
           "wall_s": round(t_end - t_begin, 2), "window_s": round(max(0.0, hi - lo), 2),
           # defined without chunk timing: all completion tokens of the block over its wall time (ramp and tail in)
           "block_tok_s": round(sum(r.get("completion_tokens") or 0 for r in ok) / (t_end - t_begin), 2)}
    if not exact(ok):
        res["aggregate_tok_s"] = None
        res["aggregate_note"] = ("no per-chunk usage (continuous_usage_stats): token arrival times unknown, window "
                                 "throughput not computed; see block_tok_s")
    elif hi - lo > 1:
        res["aggregate_tok_s"] = round(tokens_in(ok, lo, hi) / (hi - lo), 2)
        bins, w = [], a.bin_s
        t = lo
        while t + w <= hi + 1e-9:
            bins.append(tokens_in(ok, t, t + w) / w)
            t += w
        res["aggregate_bins"] = mean_ci(bins)
    valid = [r for r in ok if r.get("valid_decode")]
    res["per_stream"] = {"median_tok_s": round(statistics.median(r["tok_s"] for r in valid), 2) if valid else None,
                         "gmean_ci": gmean_ci([r["tok_s"] for r in valid]),
                         "p10_tok_s": pctl([r["tok_s"] for r in valid], 0.1)}
    tt = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    res["ttft"] = {"median_ms": round(statistics.median(tt) * 1000, 1) if tt else None,
                   "p90_ms": round(pctl(tt, 0.9) * 1000, 1) if tt else None}
    st = [r["step_ms"] for r in valid if r.get("step_ms") and r.get("steps_source") == "response"]
    if st:
        res["per_request_step_ms"] = gmean_ci(st)
        res["per_request_tokens_per_step"] = gmean_ci([r["tokens_per_step"] for r in valid if r.get("tokens_per_step")])
    res["engine"] = block_decomposition(engines.delta(before, after), t_end - t_begin)
    res["_window"] = (t_begin, t_end, lo, hi)
    return res, recs


def split_by_state(series, spans):
    """{state: [tokens, seconds]} of long-session decoding, by what the short requests were doing.

    series: per long request [(time, cumulative tokens)] at its events; spans: per short request (send, first
    token, end). The interval between two long-session events is split at every change of the short requests'
    state; its tokens are spread over the parts in proportion to time (an assumption: only event times are seen)."""
    by = {"alone": [0.0, 0.0], "short_prefilling": [0.0, 0.0], "short_decoding": [0.0, 0.0]}
    edges = sorted({t for sp in spans for t in sp})

    def state(t):
        if any(s0 <= t < s1 for s0, s1, _ in spans):
            return "short_prefilling"
        if any(s1 <= t < s2 for _, s1, s2 in spans):
            return "short_decoding"
        return "alone"
    for ct in series:
        for (ta, ca), (tb, cb) in zip(ct, ct[1:]):
            if tb <= ta:
                continue
            cuts = [ta] + [e for e in edges if ta < e < tb] + [tb]
            for u, v in zip(cuts, cuts[1:]):
                k = state((u + v) / 2)
                by[k][0] += (cb - ca) * (v - u) / (tb - ta)
                by[k][1] += v - u
    return by


# ------------------------------------------------------------------------------------------- agentic mix
def agentic_mix(arm, a, prompts):
    def chat_tokens(msgs):
        r = R.one(arm, {"messages": msgs}, a, 1)
        return r.get("prompt_tokens") if r.get("ok") else None
    cpt, method = longdoc.calibrate(arm.url, arm.model, arm.headers, chat_tokens)
    doc, _, _ = longdoc.sized(arm.url, arm.model, a.long_tokens, cpt, seed=1, salt="mix %s" % uuid.uuid4().hex,
                              headers=arm.headers)
    log(a, "%s mix: long document %d chars (~%d tokens at %.2f chars/token, %s)" % (
        arm.label, len(doc), a.long_tokens, cpt, method))
    before = engines.scrape(arm.url, arm.headers) if a.metrics else None
    lock = threading.Lock()
    longs, shorts, dropped = [], [], [0]
    t0 = time.perf_counter()
    stop = t0 + a.mix_duration

    def long_session():
        turn = 0
        while time.perf_counter() < stop:
            r = R.derive(R.one(arm, {"messages": longdoc.messages(doc, longdoc.QUESTIONS[turn % len(longdoc.QUESTIONS)])},
                               a, a.long_max_tokens))
            r.update({"class": "long", "turn": turn})
            with lock:
                longs.append(r)
            log(a, "%s mix: long turn %d: %s" % (arm.label, turn, "ttft %.1f s, %s tok/s, %s cached" % (
                r.get("ttft_s") or 0, round(r["tok_s"], 1) if r.get("tok_s") else "-", r.get("cached_tokens"))
                if r.get("ok") else r.get("error")))
            turn += 1

    active = [0]

    def short(i):
        p = prompts[i % len(prompts)]
        r = R.derive(R.one(arm, p, a, a.short_max_tokens))
        r.update({"class": "short", "id": p["id"], "category": p["category"]})
        with lock:
            shorts.append(r)
            active[0] -= 1

    lt = threading.Thread(target=long_session)
    lt.start()
    rng = random.Random(a.seed)
    threads = []
    t = t0 + a.short_delay
    i = 0
    while True:
        t += rng.expovariate(1.0 / a.short_every)
        if t >= stop:
            break
        time.sleep(max(0.0, t - time.perf_counter()))
        with lock:
            if active[0] >= a.short_max_conc:
                dropped[0] += 1
                continue
            active[0] += 1
        th = threading.Thread(target=short, args=(i,))
        th.start()
        threads.append(th)
        i += 1
    for th in threads:
        th.join()
    lt.join()
    t_end = time.perf_counter()
    after = engines.scrape(arm.url, arm.headers) if a.metrics else None
    sok = [r for r in shorts if r.get("ok")]
    spans = [(r["t_send"], r["t_send"] + r["ttft_s"], r["t_send"] + r["total_s"]) for r in sok
             if r.get("ttft_s") is not None]
    lok_exact = exact([r for r in longs if r.get("ok")])
    by = split_by_state([chunk_times(r) for r in longs if r.get("ok")], spans) if lok_exact else {}
    lok = [r for r in longs if r.get("ok")]
    res = {"long_prompt_tokens": lok[0].get("prompt_tokens") if lok else None, "chars_per_token": round(cpt, 3),
           "calibration": method, "duration_s": a.mix_duration, "wall_s": round(t_end - t0, 2),
           "long_turns": [{"turn": r["turn"], "ok": r.get("ok"), "ttft_s": _r3(r.get("ttft_s")),
                           "cached_tokens": r.get("cached_tokens"), "prompt_tokens": r.get("prompt_tokens"),
                           "tok_s": _r3(r.get("tok_s")), "completion_tokens": r.get("completion_tokens")}
                          for r in longs],
           "long_by_state": {k: {"tokens": round(v[0], 1), "seconds": round(v[1], 3),
                                 "tok_s": round(v[0] / v[1], 2) if v[1] > 0.5 else None} for k, v in by.items()}
           if lok_exact else None,
           "long_by_state_note": None if lok_exact else "no per-chunk usage: the split by state needs token counts per event",
           "long": {"median_tok_s": _med([r.get("tok_s") for r in lok if r.get("valid_decode")]),
                    "ttft_cold_s": _r3(lok[0].get("ttft_s")) if lok else None,
                    "median_ttft_cached_s": _med([r.get("ttft_s") for r in lok[1:]], 3)},
           "short": {"n": len(shorts), "failed": len(shorts) - len(sok), "dropped_at_cap": dropped[0],
                     "median_ttft_ms": _med([r["ttft_s"] * 1000 for r in sok if r.get("ttft_s") is not None]),
                     "p90_ttft_ms": round(pctl([r["ttft_s"] * 1000 for r in sok if r.get("ttft_s")], 0.9) or 0, 1)
                     if sok else None,
                     "median_tok_s": _med([r.get("tok_s") for r in sok if r.get("valid_decode")]),
                     "gmean_ci": gmean_ci([r.get("tok_s") for r in sok if r.get("valid_decode")])},
           "engine": block_decomposition(engines.delta(before, after), t_end - t0)}
    return res, longs + shorts


def _r3(x):
    return None if x is None else round(x, 3)


def _med(v, nd=2):
    v = [x for x in v if x is not None]
    return round(statistics.median(v), nd) if v else None


def strip(recs, keep_chunks):
    out = []
    for r in recs:
        r = dict(r)
        r.pop("text", None)
        r.pop("t_send", None)
        if not keep_chunks:
            r.pop("chunks", None)
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="speedbench load", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap, "speedbench-load")
    ap.add_argument("--conc", default="1,2,4,8", help="concurrency levels (comma list); '' with --mix skips them")
    ap.add_argument("--duration", type=float, default=60, help="seconds per level (and of the mix)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--bin-s", type=float, default=10, help="bin width of the aggregate-throughput CI")
    ap.add_argument("--mix", choices=["agentic"], default=None, help="add the agentic mix after the levels")
    ap.add_argument("--mix-duration", type=float, default=180, help="seconds of the agentic mix")
    ap.add_argument("--long-tokens", type=int, default=32768, help="mix: long-session document size in tokens")
    ap.add_argument("--long-max-tokens", type=int, default=512)
    ap.add_argument("--short-every", type=float, default=8.0, help="mix: mean seconds between short arrivals")
    ap.add_argument("--short-delay", type=float, default=5.0, help="mix: seconds before the first arrival may come")
    ap.add_argument("--short-max-tokens", type=int, default=160)
    ap.add_argument("--short-max-conc", type=int, default=6, help="mix: arrivals above this many in flight are dropped")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--warmup-prefill", type=int, default=None,
                    help="tokens of the long cold warm-up prefill (default 8192; with --mix the long document size)")
    ap.add_argument("--keep-chunks", action="store_true")
    a = ap.parse_args(argv)
    levels = a.levels = [int(x) for x in a.conc.split(",") if x.strip()]
    if not levels and not a.mix:
        raise SystemExit("nothing to do: give --conc levels and/or --mix agentic")
    if a.warmup_prefill is None:
        a.warmup_prefill = a.long_tokens if a.mix else 8192
    a.t0 = t0 = time.time()
    arm, counters = make_arm(a, "load")
    body, sha = P.load()
    prompts = short_prompts(body)
    sampler = power.start(a)
    try:
        doc = _load_body(a, arm, body, sha, prompts, t0, sampler)
    finally:
        if sampler is not None:
            time.sleep(1.0)
            sampler.stop()
    if sampler is not None:
        for lv, recs in doc.pop("_level_recs"):
            level_energy(sampler, lv, [r for r in recs if r.get("ok")])
        doc["energy"] = sampler.describe(keep_samples=not a.power_no_samples)
        doc["energy"].update({"note": power.NOTE[a.power_kind], "idle_w": doc.pop("_idle_w"), "per_request": False,
                              "kind": a.power_kind})
        if doc.get("_mix_window"):
            t_a, t_b, tok = doc.pop("_mix_window")
            e = sampler.energy(t_a, t_b)
            if e and tok:
                doc["mix"]["energy"] = {"energy_j": round(e, 1), "j_per_tok": round(e / tok, 5)}
    else:
        for lv, _ in doc.pop("_level_recs"):
            lv.pop("_window", None)
    for k in ("_idle_w", "_mix_window"):
        doc.pop(k, None)
    doc["elapsed_s"] = round(time.time() - t0, 1)
    doc["meta"] = M.collect(a, arm.meta, counters, mode="load" + ("+mix" if a.mix else ""))
    path = a.out or "speedbench-load-%s-%s.json" % (arm.label, time.strftime("%Y%m%d-%H%M%S"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    if a.registry:
        from . import registry
        registry.append(a.registry, doc, path)
    from . import report
    report.print_load(doc)
    print("wrote %s (%.1f min)" % (path, (time.time() - t0) / 60))
    return path


def _load_body(a, arm, body, sha, prompts, t0, sampler):
    R.prepare(arm, a, ())
    idle_w = power.idle(sampler, a.power_idle)
    doc = {"schema": "speedbench-load/1", "tool_version": __version__, "label": arm.label, "block": a.block,
           "run_id": uuid.uuid4().hex, "started": datetime.datetime.fromtimestamp(t0).astimezone().isoformat(
               timespec="seconds"),
           "prompt_set": {"version": body.get("version"), "sha256": sha, "n_selected": len(prompts)},
           "settings": {"conc": a.levels, "duration": a.duration, "max_tokens": a.max_tokens, "effort": a.effort,
                        "effort_field": a.effort_field, "mix": a.mix, "mix_duration": a.mix_duration, "long_tokens": a.long_tokens,
                        "long_max_tokens": a.long_max_tokens, "short_every": a.short_every,
                        "short_max_tokens": a.short_max_tokens, "warmup_prefill": a.warmup_prefill,
                        "metrics": a.metrics, "extra_body": arm.extra},
           "arm": {"url": arm.url, "model": arm.model, "engine": arm.engine, "continuous_usage": arm.continuous_usage},
           "server": arm.meta, "levels": [], "requests": []}
    doc["session_id"] = doc["run_id"]
    doc["_level_recs"], doc["_idle_w"] = [], idle_w
    if a.power_cmd:
        doc["settings"]["power"] = {"cmds": a.power_cmd, "kind": a.power_kind}
    for i, c in enumerate(a.levels):
        log(a, "%s: c=%d for %.0f s ..." % (arm.label, c, a.duration))
        res, recs = level(arm, a, c, prompts, offset=17 * i)
        doc["levels"].append(res)
        doc["_level_recs"].append((res, recs))
        doc["requests"] += strip(recs, a.keep_chunks)
        log(a, "%s: c=%d aggregate %s tok/s, per-stream median %s tok/s, TTFT median %s ms, %d requests" % (
            arm.label, c, res.get("aggregate_tok_s"), res["per_stream"]["median_tok_s"], res["ttft"]["median_ms"],
            res["requests"]))
    if a.mix:
        t_mix = time.perf_counter()
        res, recs = agentic_mix(arm, a, prompts)
        doc["mix"] = res
        doc["_mix_window"] = (t_mix, time.perf_counter(),
                              sum(r.get("completion_tokens") or 0 for r in recs if r.get("ok")))
        doc["requests"] += strip(recs, a.keep_chunks)
    return doc


if __name__ == "__main__":
    main()
