"""speedbench longctx: prefill/TTFT curve and decode speed at long contexts, cold and prefix-cached.

For every context size (default 8K, 16K, 32K, 64K, 128K tokens of the served model) and repetition:
  cold    a generated document with a unique salt on its first line (no prefix cache can serve it) + a question;
  cached  the same document + a different question (the cache serves the document; only the tail is computed).
At the decode contexts (default 32K, 64K, 128K) both requests decode --decode-tokens tokens and give decode tok/s
(and the engine decomposition with --metrics); elsewhere they stop after a few tokens (TTFT only).
Documents are deterministic (seeded generators of the prompt set's long documents); their size in tokens is
calibrated once per run with /tokenize or, without it, two 1-token requests. Before anything is timed, the
warm-up runs one cold prefill of the largest context (the first long prefill after a boot runs on cold kernels).
"""
import argparse
import datetime
import json
import statistics
import time
import uuid

from . import __version__, longdoc, meta as M, run as R
from .load import common_args, gmean_ci, log, make_arm


def parse_sizes(s):
    out = []
    for x in s.split(","):
        x = x.strip().lower()
        if not x:
            continue
        out.append(int(float(x[:-1]) * 1024) if x.endswith("k") else int(x))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="speedbench longctx", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap, "speedbench-longctx")
    ap.add_argument("--contexts", default="8k,16k,32k,64k,128k", help="prefill/TTFT curve sizes (tokens; k = 1024)")
    ap.add_argument("--decode-contexts", default="32k,64k,128k", help="sizes where decode speed is measured too")
    ap.add_argument("--reps", type=int, default=2, help="documents per size (each cold + cached)")
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--ttft-tokens", type=int, default=4, help="max_tokens of TTFT-only requests")
    ap.add_argument("--no-cached", action="store_true", help="skip the prefix-cached request")
    ap.add_argument("--max-context", type=int, default=None,
                    help="skip sizes above this (default: the server's max_model_len when it reports one)")
    ap.add_argument("--warmup-prefill", type=int, default=None, help="long warm-up prefill tokens (default: largest size)")
    a = ap.parse_args(argv)
    sizes = sorted(set(parse_sizes(a.contexts) + parse_sizes(a.decode_contexts)))
    dec = set(parse_sizes(a.decode_contexts))
    a.t0 = t0 = time.time()
    arm, counters = make_arm(a, "longctx")
    limit = a.max_context
    if limit is None:
        try:
            limit = int(arm.meta["models"]["data"][0].get("max_model_len") or 0) or None
        except (KeyError, IndexError, TypeError, ValueError):
            limit = None

    def chat_tokens(msgs):
        r = R.one(arm, {"messages": msgs}, a, 1)
        return r.get("prompt_tokens") if r.get("ok") else None
    cpt, method = longdoc.calibrate(arm.url, arm.model, arm.headers, chat_tokens)
    log(a, "%s: %.3f chars per token (%s)" % (arm.label, cpt, method))
    runnable = [n for n in sizes if not limit or n + a.decode_tokens + 64 <= limit]
    skipped = [n for n in sizes if n not in runnable]
    if a.warmup_prefill is None:
        a.warmup_prefill = max(runnable) if runnable else 8192
    R.prepare(arm, a, (), cpt=cpt)
    nonce = uuid.uuid4().hex[:12]
    recs = []
    for n in runnable:
        for rep in range(a.reps):
            q_cold = longdoc.QUESTIONS[rep % len(longdoc.QUESTIONS)]
            q_hot = longdoc.QUESTIONS[(rep + 3) % len(longdoc.QUESTIONS)]
            doc, counted, conv = longdoc.sized(arm.url, arm.model, n, cpt, seed=rep,
                                               salt="%s-%d-%d" % (nonce, n, rep), question=q_cold, headers=arm.headers)
            mt = a.decode_tokens if n in dec else a.ttft_tokens
            if counted and not conv:
                log(a, "%s: %d rep %d: sizing did not converge (%d tokens)" % (arm.label, n, rep, counted))
            if limit and counted and counted + mt + 64 > limit:
                recs.append({"context": n, "rep": rep, "kind": "skipped", "ok": False, "tokenized": counted,
                             "error": "prompt of %d tokens + %d output exceeds the context limit %d" % (
                                 counted, mt, limit)})
                continue
            for kind, q in (("cold", q_cold), ("cached", q_hot)):
                if kind == "cached" and a.no_cached:
                    continue
                r = R.derive(R.one(arm, {"messages": longdoc.messages(doc, q)}, a, mt, metrics=a.metrics))
                r.pop("text", None)
                r.pop("t_send", None)
                r.pop("chunks", None)
                r.update({"context": n, "rep": rep, "kind": kind, "max_tokens": mt, "tokenized": counted,
                          "sizing_converged": conv})
                recs.append(r)
                log(a, "%s: %6d %-6s rep %d: %s" % (arm.label, n, kind, rep, (
                    "%d prompt tokens (%s cached), TTFT %.2f s%s" % (
                        r.get("prompt_tokens") or 0, r.get("cached_tokens"), r.get("ttft_s") or 0,
                        ", %.1f tok/s" % r["tok_s"] if r.get("tok_s") and r.get("valid_decode") else ""))
                    if r.get("ok") else "FAILED " + str(r.get("error"))))
    table = []
    for n in runnable:
        rs = [r for r in recs if r["context"] == n and r.get("ok")]
        cold = [r for r in rs if r["kind"] == "cold"]
        hot = [r for r in rs if r["kind"] == "cached"]
        row = {"context": n, "n_ok": len(rs), "n_failed": sum(1 for r in recs if r["context"] == n and not r.get("ok")),
               "prompt_tokens": _med([r.get("prompt_tokens") for r in cold], 0),
               "ttft_cold_s": _med([r.get("ttft_s") for r in cold]),
               "ttft_cached_s": _med([r.get("ttft_s") for r in hot]),
               "cached_tokens": _med([r.get("cached_tokens") for r in hot], 0)}
        if row["prompt_tokens"] and row["ttft_cold_s"]:
            row["prefill_tok_s"] = round(row["prompt_tokens"] / row["ttft_cold_s"], 1)
        elif cold:
            row["note"] = "usage.prompt_tokens missing: prefill rate unavailable"
        if n in dec:
            v = [r for r in rs if r.get("valid_decode")]
            g = gmean_ci([r["tok_s"] for r in v])
            row["decode_tok_s"] = g["gmean"] if g else None
            row["decode_ci"] = g["ci"] if g else None
            row["step_ms"] = _med([r.get("step_ms") for r in v if r.get("steps_source") in ("metrics", "response")])
            row["tokens_per_step"] = _med([r.get("tokens_per_step") for r in v
                                           if r.get("steps_source") in ("metrics", "response")])
            row["step_ms_server"] = _med([r.get("step_ms_server") for r in v])
        table.append(row)
    doc = {"schema": "speedbench-longctx/1", "tool_version": __version__, "label": arm.label, "block": a.block,
           "run_id": uuid.uuid4().hex,
           "started": datetime.datetime.fromtimestamp(t0).astimezone().isoformat(timespec="seconds"),
           "elapsed_s": round(time.time() - t0, 1),
           "settings": {"contexts": sizes, "decode_contexts": sorted(dec), "reps": a.reps,
                        "decode_tokens": a.decode_tokens, "cached": not a.no_cached, "effort": a.effort,
                        "effort_field": a.effort_field, "warmup_prefill": a.warmup_prefill, "metrics": a.metrics,
                        "extra_body": arm.extra, "chars_per_token": round(cpt, 4), "calibration": method,
                        "skipped_over_limit": skipped, "max_context": limit, "salt_nonce": nonce},
           "arm": {"url": arm.url, "model": arm.model, "engine": arm.engine, "continuous_usage": arm.continuous_usage},
           "server": arm.meta, "table": table, "requests": recs}
    doc["session_id"] = doc["run_id"]
    doc["meta"] = M.collect(a, arm.meta, counters, mode="longctx")
    path = a.out or "speedbench-longctx-%s-%s.json" % (arm.label, time.strftime("%Y%m%d-%H%M%S"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    if a.registry:
        from . import registry
        registry.append(a.registry, doc, path)
    from . import report
    report.print_longctx(doc)
    print("wrote %s (%.1f min)" % (path, (time.time() - t0) / 60))
    return path


def _med(v, nd=3):
    v = [x for x in v if x is not None]
    return round(statistics.median(v), nd) if v else None


if __name__ == "__main__":
    main()
