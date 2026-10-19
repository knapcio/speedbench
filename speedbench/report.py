"""speedbench report: print one result file of any mode (decode, load, longctx), with the speed decomposition.

  python3 -m speedbench report RESULT.json [...]

For a decode run (full/quick) it prints, per category, the geometric means and 95 % t-CIs over prompts of decode
tok/s, step time, tokens per step, draft acceptance and drafts per step (tok/s = tokens per step / step time, per
request). These are levels of one run; A/B effects come from `compare`.
"""
import argparse
import json
import sys

from .load import gmean_ci

import math
import statistics

FIELDS = (("tok_s", "tok/s", 1), ("step_ms_server", "srv step ms", 2), ("tokens_per_step", "tok/step", 3),
          ("step_ms", "client step ms*", 2),
          ("accept_rate", "accept", 3), ("draft_per_step", "drafts/step", 2))


def per_prompt(rs, f):
    """one value per prompt: the geometric mean of its requests (repeats, variants and seeds of one prompt are
    not independent samples), so the CI is over prompts"""
    by = {}
    for r in rs:
        v = r.get(f)
        if v is not None and v > 0:
            by.setdefault(r["id"], []).append(math.log(v))
    return [math.exp(statistics.fmean(v)) for v in by.values()]


def _ci(g, nd):
    if not g:
        return "-"
    if not g.get("ci"):
        return ("%." + str(nd) + "f") % g["gmean"]
    f = "%." + str(nd) + "f"
    return (f + " [" + f + ", " + f + "]") % (g["gmean"], g["ci"][0], g["ci"][1])


def print_decode(doc, w=sys.stdout.write):
    reqs = [r for r in doc["requests"] if r.get("kind") == "measure" and r.get("ok") and r.get("valid_decode")]
    m = doc.get("meta") or {}
    w("%s  [%s, %s %s, %s]  mode %s, %d valid requests\n" % (
        doc["label"], m.get("model") or doc["arm"].get("model"), m.get("engine") or doc["arm"].get("engine"),
        m.get("engine_version") or "", m.get("quant") or "", doc["settings"]["mode"], len(reqs)))
    srcs = sorted({r.get("steps_source") or "-" for r in reqs})
    w("steps from %s%s\n" % ("/".join(srcs), "" if set(srcs) <= {"metrics", "response"} else
                             " (stream events: valid only if the server emits one event per engine step)"))
    present = [f for f in FIELDS if any(r.get(f[0]) is not None for r in reqs)]
    w("%-13s %4s  %s\n" % ("category", "n", "".join("%-26s" % f[1] for f in present)))
    cats = sorted({r["category"] for r in reqs}) + ["ALL"]
    for c in cats:
        rs = [r for r in reqs if c == "ALL" or r["category"] == c]
        w("%-13s %4d  %s\n" % (c, len({r["id"] for r in rs}), "".join(
            "%-26s" % _ci(gmean_ci(per_prompt(rs, f)), nd) for f, _, nd in present)))
    w("(n = prompts; geometric mean [95 % CI over prompts, repeats/variants averaged within a prompt];\n"
      " tok/s = tok/step / client step time per request; * client step time infers the steps inside the client\n"
      " interval from tokens: an approximation, the server step time is the engine's decode time per step)\n")
    s = doc.get("summary") or {}
    ag = s.get("agent")
    if ag:
        w("agent turns (medians): %-5s %10s %10s %10s %10s %10s %10s\n" % (
            "turn", "TTFT ms", "prompt tok", "cached", "new tok", "tok/s", "e2e s"))
        for t, b in list(ag["per_turn"].items()) + [("2+", ag["cached"])]:
            w("                       %-5s %10s %10s %10s %10s %10s %10s\n" % (
                t, b.get("median_ttft_ms"), b.get("median_prompt_tokens"), b.get("median_cached_tokens"),
                b.get("median_new_tokens"), b.get("median_tok_s"), b.get("median_e2e_s")))
        w("cache hit vs an ideal cache (previous request's prompt) on turns 2+: %s\ncache isolation: %s\n" % (
            ag["cached"].get("cache_hit_vs_ideal", "n/a (engine reports no cached tokens)"), ag.get("isolation")))
    en = doc.get("energy")
    if en:
        w("energy (%s): idle %s W%s\n" % (en.get("note"), en.get("idle_w"), "; " + en["warning"] if en.get("warning") else ""))
        for c in sorted(k for k in s if isinstance(s.get(k), dict) and s[k].get("energy")):
            e = s[c]["energy"]
            w("  %-13s %.4f J/token decode (%.2f tok/J), %s W while decoding%s\n" % (
                c, e["decode_j_per_tok"], e["decode_tok_per_j"], e.get("decode_mean_w"),
                ", request %.4f J/token" % e["request_j_per_tok"] if e.get("request_j_per_tok") else ""))
        if s.get("energy_phase_j_per_tok"):
            w("  whole measured phase (gaps and priming included): %.4f J per output token\n" % s["energy_phase_j_per_tok"])
    gd = doc.get("guard")
    if gd:
        failed = [r["id"] for r in gd["items"] if not r["pass"]]
        w("quality guard: %d/%d passed%s%s\n" % (gd["passed"], gd["total"], ("; failed: " + ", ".join(failed)) if failed else "",
                                                 "; %d without an answer within max_tokens" % gd["truncated"]
                                                 if gd.get("truncated") else ""))


def print_load(doc, w=sys.stdout.write):
    w("%s  load: %s\n" % (doc["label"], ", ".join("c=%d" % c for c in doc["settings"]["conc"]) or "mix only"))
    if doc.get("levels"):
        w("%-5s %-26s %-24s %-11s %-11s %s\n" % ("c", "aggregate tok/s [CI bins]", "per-stream tok/s [CI]",
                                                  "TTFT med", "TTFT p90", "engine (stream tok/step, srv step ms, batch)"))
    for lv in doc.get("levels") or []:
        b = lv.get("aggregate_bins")
        agg = ("%s" % lv.get("aggregate_tok_s") + (" [%.1f, %.1f]" % tuple(b["ci"]) if b and b.get("ci") else "")
               if lv.get("aggregate_tok_s") is not None else "n/a (block %s)" % lv.get("block_tok_s"))
        ps = lv["per_stream"]["gmean_ci"]
        e = lv.get("engine") or {}
        eng = ", ".join(str(x) for x in (e.get("tokens_per_stream_step"), e.get("stream_step_ms_server"),
                                          e.get("batch_occupancy")) if x is not None) or "-"
        w("%-5d %-26s %-24s %-11s %-11s %s\n" % (lv["conc"], agg, _ci(ps, 1), lv["ttft"]["median_ms"],
                                                  lv["ttft"]["p90_ms"], eng))
        if lv.get("energy"):
            e = lv["energy"]
            w("      energy: %s J per completion token over the block (%s tok/J), %s W mean%s\n" % (
                e.get("block_j_per_tok"), e.get("block_tok_per_j"), e.get("mean_w"),
                ", %s J/token in the steady window" % e["window_j_per_tok"] if e.get("window_j_per_tok") else ""))
    en = doc.get("energy")
    if en:
        w("energy (%s): idle %s W%s\n" % (en.get("note") or "source kind %s" % en.get("kind"), en.get("idle_w"),
                                         "; WARNING: " + en["warning"] if en.get("warning") else ""))
        for src in en.get("sources") or []:
            if src.get("error"):
                w("  WARNING power source %r: %s\n" % (src["cmd"], src["error"]))
    mx = doc.get("mix")
    if mx:
        w("agentic mix: long session %s prompt tokens; cold TTFT %s s, cached TTFT median %s s, decode median %s tok/s\n"
          % (mx.get("long_prompt_tokens"), mx["long"].get("ttft_cold_s"), mx["long"].get("median_ttft_cached_s"),
             mx["long"].get("median_tok_s")))
        if mx.get("long_by_state_note"):
            w("  %s\n" % mx["long_by_state_note"])
        for k, v in (mx.get("long_by_state") or {}).items():
            w("  long decode while short requests %-18s %s tok/s over %.1f s\n" % (
                {"alone": "are absent:", "short_prefilling": "wait/prefill:", "short_decoding": "decode:"}[k],
                v.get("tok_s"), v.get("seconds")))
        s = mx["short"]
        w("  short requests: %d (%d failed, %d dropped at cap), TTFT median %s ms / p90 %s ms, decode median %s tok/s\n"
          % (s["n"], s["failed"], s["dropped_at_cap"], s["median_ttft_ms"], s["p90_ttft_ms"], s["median_tok_s"]))


def print_longctx(doc, w=sys.stdout.write):
    st = doc["settings"]
    w("%s  longctx: %.3f chars/token (%s)%s\n" % (doc["label"], st["chars_per_token"], st["calibration"],
                                                   ("; skipped above the limit: %s" % st["skipped_over_limit"])
                                                   if st.get("skipped_over_limit") else ""))
    w("%-8s %-8s %-10s %-11s %-12s %-9s %-24s %-9s %s\n" % ("context", "tokens", "TTFT cold", "prefill t/s",
                                                            "TTFT cached", "cached", "decode tok/s [CI]", "step ms",
                                                            "tok/step"))
    for r in doc["table"]:
        dec = "-"
        if r.get("decode_tok_s"):
            dec = "%.1f" % r["decode_tok_s"] + (" [%.1f, %.1f]" % tuple(r["decode_ci"]) if r.get("decode_ci") else "")
        w("%-8s %-8s %-10s %-11s %-12s %-9s %-24s %-9s %s\n" % (
            "%dK" % (r["context"] // 1024), r.get("prompt_tokens"), r.get("ttft_cold_s"), r.get("prefill_tok_s"),
            r.get("ttft_cached_s"), r.get("cached_tokens"), dec, r.get("step_ms") or "-", r.get("tokens_per_step") or "-"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="speedbench report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    for i, p in enumerate(a.files):
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        if i:
            print()
        sch = doc.get("schema")
        if sch == "speedbench-result/1":
            print_decode(doc)
        elif sch == "speedbench-load/1":
            print_load(doc)
        elif sch == "speedbench-longctx/1":
            print_longctx(doc)
        else:
            raise SystemExit("%s: unknown result schema %r" % (p, sch))


if __name__ == "__main__":
    main()
