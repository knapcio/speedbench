"""speedbench registry: a JSONL log of runs (metadata + headline numbers) and the README results tables.

  python3 -m speedbench registry add RESULT.json [...] [--registry results/registry.jsonl]
  python3 -m speedbench registry table [--registry results/registry.jsonl] [--readme README.md]

Every run command takes `--registry PATH` to append itself when it finishes. `table` rewrites the part of the
README between the markers below (or prints it). Rows are single runs and descriptive: an A/B claim needs
`compare` on interleaved runs, never two rows of this table (README "Method").
"""
import argparse
import hashlib
import json
import os
import sys

BEGIN = "<!-- speedbench:results:begin -->"
END = "<!-- speedbench:results:end -->"
DEFAULT = os.path.join("results", "registry.jsonl")
DECODE_CATS = ("prose", "code", "structured", "agentic", "multilingual", "longctx")


def _r(x, nd=1):
    return None if x is None else round(x, nd)


def headline(doc):
    sch = doc.get("schema", "")
    s = doc.get("summary") or {}
    if sch == "speedbench-result/1":
        al = s.get("ALL") or {}
        h = {"n": al.get("n"), "failed": s.get("failed"), "geomean_tok_s": al.get("geomean_tok_s"),
             "median_ttft_ms": al.get("median_ttft_ms"), "median_step_ms": al.get("median_step_ms"),
             "tokens_per_step": al.get("median_tokens_per_step"),
             "max_tokens": (doc.get("settings") or {}).get("max_tokens"),
             "prompt_set": (doc.get("prompt_set") or {}).get("version")}
        for c in DECODE_CATS:
            if (s.get(c) or {}).get("geomean_tok_s"):
                h[c] = s[c]["geomean_tok_s"]
        if al.get("energy"):
            h["decode_tok_per_j"] = al["energy"]["decode_tok_per_j"]
            h["power_kind"] = (doc.get("energy") or {}).get("kind")
        ag = s.get("agent")
        if ag:
            h["agent"] = {"ttft_cold_ms": ag["cold"].get("median_ttft_ms"), "ttft_cached_ms": ag["cached"].get("median_ttft_ms"),
                          "cache_hit": ag["cached"].get("cache_hit_vs_ideal"), "e2e_s": ag["cached"].get("median_e2e_s"),
                          "prompt_tokens_cold": ag["cold"].get("median_prompt_tokens"),
                          "new_tokens_cached": ag["cached"].get("median_new_tokens")}
        if doc.get("guard"):
            h["guard"] = "%d/%d" % (doc["guard"]["passed"], doc["guard"]["total"])
        return h
    if sch == "speedbench-load/1":
        h = {"levels": {}}
        for lv in doc.get("levels") or []:
            h["levels"][str(lv["conc"])] = {"aggregate_tok_s": lv.get("aggregate_tok_s"),
                                            "per_stream_tok_s": lv.get("per_stream", {}).get("median_tok_s"),
                                            "median_ttft_ms": lv.get("ttft", {}).get("median_ms")}
        if doc.get("mix"):
            m = doc["mix"]
            h["mix"] = {"long_tokens": m.get("long_prompt_tokens"),
                        "long_alone_tok_s": (m.get("long_by_state") or {}).get("alone", {}).get("tok_s"),
                        "long_with_short_tok_s": (m.get("long_by_state") or {}).get("short_decoding", {}).get("tok_s"),
                        "long_during_prefill_tok_s": (m.get("long_by_state") or {}).get("short_prefilling", {}).get("tok_s"),
                        "short_ttft_ms": (m.get("short") or {}).get("median_ttft_ms"),
                        "short_tok_s": (m.get("short") or {}).get("median_tok_s")}
        return h
    if sch == "speedbench-longctx/1":
        h = {"contexts": {}}
        for row in doc.get("table") or []:
            h["contexts"][str(row["context"])] = {k: row.get(k) for k in (
                "prompt_tokens", "ttft_cold_s", "ttft_cached_s", "prefill_tok_s", "decode_tok_s", "step_ms")}
        return h
    return {}


def entry(doc, path):
    m = doc.get("meta") or {}
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "date": m.get("date") or doc.get("started"), "label": doc.get("label"), "kind": doc.get("schema"),
        "mode": m.get("mode") or (doc.get("settings") or {}).get("mode"), "model": m.get("model"),
        "quant": m.get("quant"), "engine": m.get("engine"), "engine_version": m.get("engine_version"),
        "engine_commit": m.get("engine_commit"), "config": m.get("config"), "hardware": m.get("hardware"),
        "thinking": m.get("thinking"), "effort": m.get("effort"), "host_compacted": m.get("host_compacted"),
        "fresh_boot": m.get("fresh_boot"), "clock_policy": m.get("clock_policy"), "notes": m.get("notes"),
        "run_id": doc.get("run_id"), "session_id": doc.get("session_id"), "block": doc.get("block"),
        "result_file": os.path.relpath(os.path.abspath(path)), "result_sha256": sha,
        "tool_version": m.get("tool_version") or doc.get("tool_version"), "headline": headline(doc),
    }


def append(reg, doc, path):
    e = entry(doc, path)
    d = os.path.dirname(os.path.abspath(reg))
    os.makedirs(d, exist_ok=True)
    if os.path.exists(reg):
        for old in load(reg):
            if old.get("result_sha256") == e["result_sha256"]:
                print("registry: %s already registered" % path)
                return None
    with open(reg, "a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    print("registry: appended %s to %s" % (e["label"], reg))
    return e


def load(reg):
    out = []
    with open(reg, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _c(x):
    if x is None or x == "":
        return "–"
    if isinstance(x, float):
        return ("%.0f" % x) if abs(x) >= 1000 else ("%.1f" % x) if abs(x) >= 10 else ("%.2f" % x)
    return str(x).replace("|", "/")


def _engine(e):
    v = e.get("engine_version")
    c = e.get("engine_commit")
    tail = [x for x in (v, c) if x and x != "unknown"]
    return "%s%s" % (e.get("engine") or "?", (" " + " ".join(tail)) if tail else "")


def _host(e):
    bits = []
    if e.get("host_compacted") not in (None, "unknown"):
        bits.append("compacted " + e["host_compacted"])
    if e.get("fresh_boot") not in (None, "unknown"):
        bits.append("fresh " + e["fresh_boot"])
    if e.get("clock_policy") not in (None, "unknown"):
        bits.append(e["clock_policy"])
    return ", ".join(bits) or "–"


def _tpj(h):
    if not h.get("decode_tok_per_j"):
        return None
    return "%.2f (%s)" % (h["decode_tok_per_j"], h.get("power_kind") or "?")


def table(entries):
    w = []
    dec = [e for e in entries if e.get("kind") == "speedbench-result/1" and not e["headline"].get("agent")]
    agt = [e for e in entries if e.get("kind") == "speedbench-result/1" and e["headline"].get("agent")]
    lod = [e for e in entries if e.get("kind") == "speedbench-load/1"]
    lng = [e for e in entries if e.get("kind") == "speedbench-longctx/1"]
    w.append("_Generated by `python3 -m speedbench registry table` from `results/registry.jsonl`; single runs, "
             "descriptive. Decode tok/s = geometric mean over prompts of per-request decode tok/s (c1). "
             "An A/B claim needs `compare` on interleaved runs, not two rows of this table._\n")
    if dec:
        w.append("\n#### Decode, c1 (full / quick)\n\n")
        w.append("| date | label | model | quant | engine | hardware | thinking / effort | mode | tok/s | "
                 "prose | code | struct. | agentic | multi | longctx | TTFT ms | step ms | tok/step | tok/J | "
                 "host / clocks |\n")
        w.append("|" + "---|" * 20 + "\n")
        for e in dec:
            h = e["headline"]
            w.append("| %s |\n" % " | ".join(_c(x) for x in (
                (e.get("date") or "")[:16].replace("T", " "), e.get("label"), e.get("model"), e.get("quant"),
                _engine(e), e.get("hardware"), "%s / %s" % (e.get("thinking"), e.get("effort")),
                "%s %s" % (e.get("mode"), h.get("max_tokens") or ""), h.get("geomean_tok_s"), h.get("prose"),
                h.get("code"), h.get("structured"), h.get("agentic"), h.get("multilingual"), h.get("longctx"),
                h.get("median_ttft_ms"), h.get("median_step_ms"), h.get("tokens_per_step"), _tpj(h), _host(e))))
    if agt:
        w.append("\n#### Multi-turn agent, c1 (prefix cache; medians over turns)\n\n")
        w.append("| date | label | model | quant | engine | hardware | thinking / effort | mode | turn-1 TTFT ms | "
                 "cached-turn TTFT ms | cache hit | new tokens | decode tok/s | turn s | tok/J | host / clocks |\n")
        w.append("|" + "---|" * 16 + "\n")
        for e in agt:
            h = e["headline"]
            g = h["agent"]
            w.append("| %s |\n" % " | ".join(_c(x) for x in (
                (e.get("date") or "")[:16].replace("T", " "), e.get("label"), e.get("model"), e.get("quant"),
                _engine(e), e.get("hardware"), "%s / %s" % (e.get("thinking"), e.get("effort")),
                "%s %s" % (e.get("mode"), h.get("max_tokens") or ""), g.get("ttft_cold_ms"), g.get("ttft_cached_ms"),
                g.get("cache_hit"), g.get("new_tokens_cached"), h.get("geomean_tok_s"), g.get("e2e_s"), _tpj(h),
                _host(e))))
    if lod:
        levels = sorted({int(c) for e in lod for c in e["headline"].get("levels", {})})
        w.append("\n#### Concurrency (aggregate decode tok/s; per-stream median in brackets; median TTFT ms)\n\n")
        w.append("| date | label | model | engine | hardware | %s | agentic mix: long alone / with short / during "
                 "short prefill tok/s, short TTFT ms | host / clocks |\n" % " | ".join("c=%d" % c for c in levels))
        w.append("|" + "---|" * (7 + len(levels)) + "\n")
        for e in lod:
            h = e["headline"]
            cells = []
            for c in levels:
                lv = h.get("levels", {}).get(str(c))
                cells.append("%s (%s; %s)" % (_c(lv.get("aggregate_tok_s")), _c(lv.get("per_stream_tok_s")),
                                              _c(lv.get("median_ttft_ms"))) if lv else "–")
            mx = h.get("mix")
            mixs = ("%s / %s / %s, %s" % (_c(mx.get("long_alone_tok_s")), _c(mx.get("long_with_short_tok_s")),
                                          _c(mx.get("long_during_prefill_tok_s")), _c(mx.get("short_ttft_ms")))
                    if mx else "–")
            w.append("| %s |\n" % " | ".join([_c((e.get("date") or "")[:16].replace("T", " ")), _c(e.get("label")),
                                              _c(e.get("model")), _c(_engine(e)), _c(e.get("hardware"))] + cells +
                                             [mixs, _host(e)]))
    if lng:
        ctxs = sorted({int(c) for e in lng for c in e["headline"].get("contexts", {})})
        w.append("\n#### Long context (TTFT cold / cached in s; decode tok/s where measured)\n\n")
        w.append("| date | label | model | engine | hardware | %s | host / clocks |\n" % " | ".join(
            "%dK" % (c // 1024) for c in ctxs))
        w.append("|" + "---|" * (6 + len(ctxs)) + "\n")
        for e in lng:
            h = e["headline"]["contexts"]
            cells = []
            for c in ctxs:
                r = h.get(str(c))
                if not r:
                    cells.append("–")
                    continue
                cells.append("%s / %s%s" % (_c(r.get("ttft_cold_s")), _c(r.get("ttft_cached_s")),
                                            ("; %s tok/s" % _c(r["decode_tok_s"])) if r.get("decode_tok_s") else ""))
            w.append("| %s |\n" % " | ".join([_c((e.get("date") or "")[:16].replace("T", " ")), _c(e.get("label")),
                                              _c(e.get("model")), _c(_engine(e)), _c(e.get("hardware"))] + cells +
                                             [_host(e)]))
    if not (dec or agt or lod or lng):
        w.append("\n(no runs registered yet)\n")
    return "".join(w)


def write_readme(readme, text):
    with open(readme, encoding="utf-8") as f:
        cur = f.read()
    if BEGIN not in cur or END not in cur:
        raise SystemExit("%s has no %s ... %s markers" % (readme, BEGIN, END))
    head, rest = cur.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + BEGIN + "\n" + text + END + tail
    with open(readme, "w", encoding="utf-8") as f:
        f.write(new)
    return new != cur


def main(argv=None):
    ap = argparse.ArgumentParser(prog="speedbench registry", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add", help="append result files to the registry")
    p_add.add_argument("files", nargs="+")
    p_add.add_argument("--registry", default=DEFAULT)
    p_tab = sub.add_parser("table", help="print the results tables or write them into the README")
    p_tab.add_argument("--registry", default=DEFAULT)
    p_tab.add_argument("--readme", default=None, help="rewrite the README between the results markers")
    a = ap.parse_args(argv)
    if a.cmd == "add":
        for p in a.files:
            with open(p, encoding="utf-8") as f:
                append(a.registry, json.load(f), p)
        return None
    entries = load(a.registry) if os.path.exists(a.registry) else []
    entries.sort(key=lambda e: e.get("date") or "")
    text = table(entries)
    if a.readme:
        changed = write_readme(a.readme, text)
        print("%s: results tables %s (%d runs)" % (a.readme, "updated" if changed else "unchanged", len(entries)))
    else:
        sys.stdout.write(text)
    return text


if __name__ == "__main__":
    main()
