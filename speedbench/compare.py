"""speedbench compare: paired A/B analysis of result files (one or more runs per arm)."""
import argparse
import hashlib
import json
import math
import os
import statistics
import sys

from . import guard as G, stats as S

HORIZONS = (64, 128, 192)
CHECKED = ("mode", "max_tokens", "temperature", "top_p", "effort", "effort_field", "conc", "seeds", "variants",
           "repeats", "prime", "skip_tools", "categories", "isolate")
AGENT_MODES = ("agent", "agent-quick")
SCREEN_MODES = ("quick", "agent-quick")
TREATMENT = ("model", "extra_body")          # may differ between arms (that is the A/B), reported
DECOMP = (("step_ms", "step time (client, approx.)"), ("step_ms_server", "step time (server)"),
          ("tokens_per_step", "tokens per step"), ("accept_rate", "draft acceptance"),
          ("draft_per_step", "drafts per step"))
HOST = ("host_compacted", "fresh_boot", "clock_policy", "engine_commit", "quant", "hardware")


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") != "speedbench-result/1":
        raise SystemExit("%s: not a speedbench result" % path)
    doc["_path"] = os.path.abspath(path)
    doc["_content"] = hashlib.sha256(json.dumps(doc["requests"], sort_keys=True).encode()).hexdigest()
    return doc


def measured(doc):
    return [r for r in doc["requests"] if r.get("kind") == "measure"]


def logs(doc, value, valid=lambda r: r.get("ok") and r.get("valid_decode")):
    """{key: mean log(value(r))} over the repeats inside one run; a key failing in any repeat is dropped"""
    acc, bad = {}, set()
    for r in measured(doc):
        v = value(r) if valid(r) else None
        if v is None or v <= 0:
            bad.add(r["key"])
            continue
        acc.setdefault(r["key"], []).append(math.log(v))
    return {k: statistics.fmean(v) for k, v in acc.items() if k not in bad}


def by_boot(docs, getter):
    """runs that share a --block label ran on one boot: average them into one observation of that boot"""
    groups = {}
    for i, d in enumerate(docs):
        groups.setdefault(d.get("block") or ("#run%d" % i), []).append(getter(d))
    out = []
    for runs in groups.values():
        keys = set(runs[0]).intersection(*runs[1:])
        out.append({k: statistics.fmean(r[k] for r in runs) for k in keys})
    return out


def horizon_rate(r, h):
    """decode tok/s over the first h generated tokens, from cumulative per-chunk usage (None if unknown)"""
    ch = r.get("chunks")
    if not ch or ch[0][2] < 0:
        return None
    k1, t1 = ch[0][2], ch[0][0]
    for t, _, cum, _ in ch[1:]:
        if cum < 0:
            return None
        if cum >= h:
            return (cum - k1) / ((t - t1) / 1000) if t > t1 else None
    return None


def name(d):
    return "%s[%s]" % (d["label"], (d.get("run_id") or "")[:6])


def decode_valid(settings):
    """the one predicate for 'this request's decode is timed', shared by the analysis and the input checks.
    T>0 and agent turns: a short output (a sampled short answer, a tool call) is a sample of the workload, not a
    failure, and is timed from MIN_DECODE_SAMPLED tokens; full/quick at T=0 use run.MIN_DECODE_TOKENS (32)."""
    if settings.get("temperature") or settings.get("mode") in AGENT_MODES:
        return lambda r: bool(r.get("ok") and r.get("tok_s") and (r.get("decode_tokens") or 0) >= MIN_DECODE_SAMPLED)
    return lambda r: bool(r.get("ok") and r.get("valid_decode"))


def boundary_mode(docs, valid=None):
    valid = valid or decode_valid(docs[0].get("settings") or {})
    ok = [r for d in docs for r in measured(d) if valid(r)]
    ex = sum(r.get("boundary") == "exact" for r in ok)
    if ok and ex >= 0.95 * len(ok):
        return "exact"
    if ex == 0:
        return "assumed"
    return "mixed (%d/%d exact)" % (ex, len(ok))


def check_inputs(As, Bs, force):
    problems, notes = [], []
    seen = {}
    for d in As + Bs:
        for k in ("_path", "_content", "run_id"):
            v = d.get(k)
            if v and v in seen:
                raise SystemExit("%s and %s are the same run (%s); a run may be used once" % (
                    seen[v], d["_path"], k.strip("_")))
            if v:
                seen[v] = d["_path"]
    ref = As[0]
    for d in As + Bs:
        if d["prompt_set"]["sha256"] != ref["prompt_set"]["sha256"]:
            problems.append("%s: different prompt set" % name(d))
        for k in CHECKED:
            if d["settings"].get(k) != ref["settings"].get(k):
                problems.append("%s: setting %s=%r differs from %r" % (name(d), k, d["settings"].get(k),
                                                                         ref["settings"].get(k)))
        rt, rr = d.get("request_template") or {}, ref.get("request_template") or {}
        for k in sorted(set(rt) | set(rr)):
            if k not in TREATMENT and rt.get(k) != rr.get(k):
                problems.append("%s: effective request field %s=%r differs from %r" % (name(d), k, rt.get(k), rr.get(k)))
    def treat(d, k):
        return json.dumps((d.get("arm") or {}).get(k), sort_keys=True)
    for arm, docs in (("A", As), ("B", Bs)):
        for k in TREATMENT:
            if len({treat(d, k) for d in docs}) > 1:
                problems.append("arm %s mixes runs with different %s" % (arm, k))
    for k in TREATMENT:
        if treat(As[0], k) != treat(Bs[0], k):
            notes.append("treatment difference %s: A=%s B=%s" % (k, treat(As[0], k), treat(Bs[0], k)))
    valid = decode_valid(As[0].get("settings") or {})
    bA, bB = boundary_mode(As, valid), boundary_mode(Bs, valid)
    if bA != bB or bA.startswith("mixed"):
        problems.append("token/time boundary differs: A %s, B %s (per-chunk usage must be on or off in both)" % (bA, bB))
    for arm, docs in (("A", As), ("B", Bs)):
        blocks = [d.get("block") for d in docs]
        if len(docs) > 1 and not all(blocks):
            notes.append("arm %s: %d runs without --block labels are treated as separate boots" % (arm, len(docs)))
        for d in docs:
            exp = d["settings"].get("expected_jobs")
            got = len(measured(d))
            if exp is not None and got != exp:
                problems.append("%s: %d of %d jobs recorded (interrupted run)" % (name(d), got, exp))
    pw = {json.dumps((d.get("settings") or {}).get("power"), sort_keys=True) for d in As + Bs}
    if len(pw) > 1:
        notes.append("power sampling differs between runs (%s): polling the GPU driver can cost speed; compare speed "
                     "only between runs that sampled power the same way" % ", ".join(sorted(pw)))
    for k in HOST:
        vals = {str((d.get("meta") or {}).get(k, "unknown")) for d in As + Bs}
        if len(vals) > 1:
            notes.append("operator metadata %s differs between runs: %s" % (k, ", ".join(sorted(vals))))
    if problems and not force:
        raise SystemExit("incompatible runs (use --force to compare anyway):\n  " + "\n  ".join(problems))
    return problems, notes, bA


def session_check(As, Bs):
    """are the arms interleaved inside shared sessions? -> (shared, interleaved, messages)"""
    shared = len(As) == len(Bs) and all(x.get("session_id") and x.get("session_id") == y.get("session_id")
                                        for x, y in zip(As, Bs))
    msgs = []
    if not shared:
        sa = {d.get("session_id") for d in As}
        sb = {d.get("session_id") for d in Bs}
        msgs.append("CROSS-SESSION COMPARISON: arm A ran in %d session(s), arm B in %d other session(s). The same "
                    "bit-exact configuration has measured ~2 %% apart in two sessions a few hours apart on our "
                    "fleet, while interleaved pairs inside one session agree to ~0.1 %%. Differences below a few "
                    "per cent are not resolvable this way. Interleave both arms in one session (run --arm A=... "
                    "--arm B=...); if a lever needs a restart, alternate sessions A B A B ... and give each run "
                    "its --block, so the session SD is estimated from the data." % (len(sa), len(sb)))
        return shared, False, msgs
    interleaved = True
    for x, y in zip(As, Bs):
        tx = [r["t_wall"] for r in measured(x) if r.get("t_wall") is not None]
        ty = [r["t_wall"] for r in measured(y) if r.get("t_wall") is not None]
        if len(tx) < 4 or len(ty) < 4:
            continue
        span = max(tx + ty) - min(tx + ty)
        if span > 0 and abs(statistics.fmean(tx) - statistics.fmean(ty)) > 0.05 * span:
            interleaved = False
            msgs.append("NOT INTERLEAVED: in session %s the mean request time of A and B differs by %.0f %% of the "
                        "session; drift inside the session is then confounded with the lever" % (
                            (x.get("session_id") or "")[:8], 100 * abs(statistics.fmean(tx) - statistics.fmean(ty)) / span))
    return shared, interleaved, msgs


def fmt_ci(res):
    o = res["overall"] if "overall" in res else res
    return "%+6.2f %%  [%+.2f, %+.2f]" % (S.pct(o["est"]), S.pct(o["ci"][0]), S.pct(o["ci"][1]))


# T>0 and agent turns: a short output (a sampled short answer, a tool call) is a sample of the workload, not a failure;
# it is timed if it has at least this many decode tokens (full/quick at T=0 keep run.MIN_DECODE_TOKENS = 32)
MIN_DECODE_SAMPLED = 8


def withhold(res, chk, planned, partial, what="prompts"):
    """reason to withhold a verdict (the workload changed, or the result depends on the exclusions), or None"""
    n_exc = len(res["excluded"])
    if n_exc > max(2, math.floor(0.05 * planned)):
        return "WITHHELD: %d of %d %s excluded (failed or too short in some run); the workload changed" % (
            n_exc, planned, what)
    if partial and res["pairs_dropped"] > max(2, math.floor(0.05 * res["pairs_planned"])):
        return "WITHHELD: %d of %d pairs dropped (failed or too short in an arm); the workload changed" % (
            res["pairs_dropped"], res["pairs_planned"])
    why = S.exclusion_rule(chk, res["pairs_planned"]) if chk else None
    return ("WITHHELD: " + why) if why else None


def publishable_label(res, unit):
    """a publishable verdict requires zero outcome-dependent drops: any pair dropped because an output failed, was
    too short or could not be timed makes the verdict descriptive only (the numbers are still printed)"""
    n = res["pairs_dropped"]
    if not n:
        return ""
    return "NOT PUBLISHABLE (%d of %d pairs dropped: failed, short or untimeable outputs; descriptive only): " % (
        n, res["pairs_planned"])


def analyse(As, Bs, a, boundary="exact"):
    s0 = As[0]["settings"]
    agent_mode = s0.get("mode") in AGENT_MODES
    sampled = bool(s0.get("temperature"))     # T>0: pairs per (prompt, seed), see stats.paired
    screen = s0.get("mode") in SCREEN_MODES
    cats, cluster, turn = {}, {}, {}
    for d in As + Bs:
        for r in measured(d):
            # agent mode: the turns of a conversation are not independent (shared content, treatment effects that
            # differ by conversation, cache state): they are clustered by conversation, pairs are turns
            cats[r["key"]] = "conversations" if agent_mode else r["category"]
            cluster[r["key"]] = r.get("conv") or r["id"] if agent_mode else r["id"]
            turn[r["key"]] = r.get("turn") or 0
    partial = sampled or agent_mode
    valid_dec = decode_valid(s0)
    ok_any = (lambda r: r.get("ok"))
    unit = "conversations" if agent_mode else "prompts"
    out = {"A": [name(d) for d in As], "B": [name(d) for d in Bs], "boundary": boundary, "unit": unit,
           "partial": partial}
    prior = a.boot_prior / 100
    shared, interleaved, smsgs = session_check(As, Bs)
    out["shared_boots"] = shared
    out["interleaved"] = interleaved
    out["session_warnings"] = smsgs
    cancel = shared and interleaved          # only then does the session shock cancel in the paired difference
    out["session_cancels"] = cancel

    def inputs(value, valid=None, subset=None, **kw):
        get = lambda d: logs(d, value, valid or valid_dec)   # noqa: E731
        if cancel and "prior" not in kw:
            kw["shared_boot_sd"] = a.shared_prior / 100
        kw.setdefault("partial", partial)
        kw["alpha"] = kw.pop("alpha_override", a.alpha)
        kw["boot_prior_sd"] = kw.pop("prior", prior)
        kw["cluster"] = cluster
        cc = cats if subset is None else {k: c for k, c in cats.items() if subset(k)}
        return by_boot(As, get), by_boot(Bs, get), cc, kw

    def paired(value, valid=None, subset=None, **kw):
        A_, B_, cc, kw = inputs(value, valid, subset, **kw)
        return S.paired(A_, B_, cc, **kw)

    def checked(value, valid=None, subset=None, what=unit):
        """paired result, exclusion check and withholding reason for a verdict metric"""
        A_, B_, cc, kw = inputs(value, valid, subset)
        res = S.paired(A_, B_, cc, **kw)
        chk = S.exclusion_check(A_, B_, cc, res, kw)
        planned = len({cluster[k] for k in cc})
        return res, chk, withhold(res, chk, planned, partial, what)

    tok = lambda r: r.get("tok_s")                           # noqa: E731
    try:
        res, chk, why = checked(tok)
    except ValueError:
        res, chk, why = None, None, None
    out["decode"] = res
    out["exclusions"] = chk
    if res is None:
        out["verdict"] = "NO DECODE VERDICT: fewer than 3 %s with valid decode in every run of both arms" % unit
        out["prior_conditional"] = False
        out["boot_warnings"] = []
    else:
        out["verdict"] = why or S.verdict(res, a.alpha)
        if screen and not why:
            out["verdict"] = "SCREEN ONLY (%s mode): " % s0["mode"] + out["verdict"]
        out["prior_conditional"] = any(res["boot"][x]["source"] != "anova+prior" for x in ("A", "B"))
        if not cancel and out["prior_conditional"]:
            alt = paired(tok, prior=a.alt_prior / 100, n_boot=0)
            out["decode_boot05"] = alt["overall"]
        if not cancel and not out["verdict"].startswith("WITHHELD"):
            out["verdict"] = ("CROSS-SESSION, " if not shared else "NOT INTERLEAVED, ") + (
                "prior-conditional: " if out["prior_conditional"] else "") + out["verdict"]
        if not why:
            out["verdict"] = publishable_label(res, unit) + out["verdict"]
        out["boot_warnings"] = ["arm %s: estimated boot SD %.2f %% is more than twice the prior %.2f %%; the CI leans "
                                "on the prior: add boots or raise --boot-prior" % (
                                    x, res["boot"][x]["sigma_b_anova"] * 100, a.boot_prior)
                                for x in ("A", "B") if res["boot"][x].get("sigma_b_anova", 0) > 2 * prior]
        bA = res["boot"]["A"]
        if bA.get("sigma_session_anova") is not None:
            # interleaved sessions: the pooled session term is prior-conditional; show the data-only CI next to it
            if bA["sigma_session_anova"] > 2 * a.shared_prior / 100:
                out["boot_warnings"].append(
                    "the session-to-session SD of the paired difference, %.2f %%, is more than twice the residual "
                    "prior %.2f %%; trust the data-only CI below, or add interleaved sessions" % (
                        bA["sigma_session_anova"] * 100, a.shared_prior))
            out["decode_data_only"] = _try(lambda: paired(tok, n_boot=0, prior_df=0))
    out["ttft"] = _try(lambda: paired(lambda r: r.get("ttft_s"), ok_any, n_boot=0))
    if agent_mode:
        ag = {"isolate": s0.get("isolate")}

        def latency(value, subset=None):
            # latency is valid for every successful request, whatever its output length (short tool calls count)
            try:
                r, c, why = checked(value, ok_any, subset)
            except ValueError:
                return None, "NOT AVAILABLE: fewer than 3 conversations with successful requests in both arms"
            v = why or lower_is_better(r)
            iso = [x["summary"].get("agent", {}).get("isolation") or "UNVERIFIED" for x in As + Bs]
            bad = [i for i in iso if "FAILED" in i or "UNVERIFIED" in i]
            if bad and not why:
                v = "CACHE ISOLATION %s: " % ("FAILED" if any("FAILED" in i for i in bad) else "UNVERIFIED") + v
            if not cancel and not why:
                v = ("CROSS-SESSION, " if not shared else "NOT INTERLEAVED, ") + v
            if screen and not why:
                v = "SCREEN ONLY (%s mode): " % s0["mode"] + v
            if not why:
                v = publishable_label(r, unit) + v
            return r, v
        ag["ttft_cold"], ag["ttft_cold_verdict"] = latency(lambda r: r.get("ttft_s"), lambda k: turn.get(k) == 1)
        ag["ttft_cached"], ag["ttft_cached_verdict"] = latency(lambda r: r.get("ttft_s"), lambda k: turn.get(k, 0) > 1)
        ag["e2e"], ag["e2e_verdict"] = latency(lambda r: r.get("total_s"))
        per_turn = {}
        for t in sorted({v for v in turn.values() if v}):
            per_turn[t] = {"ttft": _try(lambda t=t: paired(lambda r: r.get("ttft_s"), ok_any,
                                                            subset=lambda k: turn.get(k) == t, n_boot=0)),
                           "tok_s": _try(lambda t=t: paired(tok, subset=lambda k: turn.get(k) == t, n_boot=0))}
        ag["per_turn"] = per_turn
        ag["summaries"] = {name(x): (x["summary"].get("agent") or {}) for x in As + Bs}
        out["agent"] = ag
    if all(d.get("energy") and (d["energy"].get("per_request")) for d in As + Bs):
        out["energy"] = _try(lambda: paired(lambda r: r.get("j_per_tok"), valid_dec, n_boot=0))
        out["energy_meta"] = {"kind": {d["energy"].get("kind") for d in As + Bs},
                              "note": As[0]["energy"].get("note"),
                              "per_arm": {nm: (d["summary"].get("ALL") or {}).get("energy") for nm, d in
                                          ((name(x), x) for x in As + Bs)},
                              "idle_w": {name(d): d["energy"].get("idle_w") for d in As + Bs}}
    if sampled and res is not None:
        out["sampling"] = _try(lambda: seed_analysis(As, Bs, res, cluster, valid_dec))
    out["guard"] = guard_sessions(As, Bs)
    if not sampled and not (agent_mode and s0.get("isolate") == "text"):
        out["divergence"] = G.divergence(As, Bs, measured)
    out["ttft_prime"] = _try(lambda: paired(lambda r: (r.get("prime") or {}).get("ttft_s"), ok_any, n_boot=0))
    srcs = {r.get("steps_source") for d in As + Bs for r in measured(d) if r.get("ok")}
    out["steps_source"] = "/".join(sorted(s for s in srcs if s))
    both = [(r["steps_stream"], r["steps_metrics"]) for d in As + Bs for r in measured(d)
            if r.get("ok") and r.get("steps_metrics") and r.get("steps_stream") is not None]
    if both:
        out["steps_agreement"] = (sum(abs(x - y) <= 2 for x, y in both), len(both))
    for field, _ in DECOMP:
        out[field] = _try(lambda f=field: paired(lambda r: r.get(f), n_boot=0))
    # equivalence (TOST at 5 %): 90 % CIs of tokens/step and drafts/step must lie inside +-margin
    for field in ("tokens_per_step", "draft_per_step"):
        out[field + "_ci90"] = _try(lambda f=field: paired(lambda r: r.get(f), n_boot=0, alpha_override=0.10))
    out["step_rule"] = step_rule(out, As, Bs, a)
    # output identity, over every measured request (all repeats, variants and runs)
    def hashes(docs):
        h = {}
        for d in docs:
            for r in measured(d):
                if r.get("ok"):
                    h.setdefault(r["key"], set()).add(r["sha256"])
        return h
    hA, hB = hashes(As), hashes(Bs)
    common = [k for k in hA if k in hB]
    out["identical_between_arms"] = (sum(hA[k] == hB[k] and len(hA[k]) == 1 for k in common), len(common))
    for nm, h, docs in (("A", hA, As), ("B", hB, Bs)):
        multi = [k for k in h if sum(1 for d in docs for r in measured(d) if r["key"] == k and r.get("ok")) > 1]
        if multi:
            out["identical_within_" + nm] = (sum(len(h[k]) == 1 for k in multi), len(multi))
    hz = {}
    for hh in HORIZONS:
        r = _try(lambda hh=hh: paired(lambda q: horizon_rate(q, hh), ok_any, n_boot=0))
        if r:
            hz[hh] = {"n": r["n_prompts"], "sd_d": r["sd_d"], "est": r["overall"]["est"]}
    out["horizons"] = hz
    out["runs"] = {name(d): {"summary": d["summary"], "block": d.get("block")} for d in As + Bs}
    return out


def step_rule(out, As, Bs, a):
    """The pre-registered step-time rule. Step time is (nearly) free of trajectory luck: a lever that only changes
    the cost of a step (kernels, communication, scheduling) moves tokens/step by luck alone, so its real speed
    effect is the step-time change and the step-time CI is several times narrower than the tok/s CI.

    The step-time verdict DECIDES only if all hold:
      1. declared before the run: `run --preregister step` (recorded in the B run; `compare --decide-on step`
         afterwards is reported as post hoc and never replaces the tok/s verdict);
      2. steps counted by the engine (response fields or /metrics), not stream events, in both arms;
      3. acceptance unchanged within the margin: the 90 % CI of tokens/step lies inside +-margin (TOST, 5 %);
      4. draft length unchanged within the margin when the engine reports draft tokens (a lever that drafts
         more per step legitimately costs more per step and must be judged on tok/s).
    Otherwise the tok/s verdict decides."""
    if out.get("decode") is None:
        return {}
    pre = {(d.get("settings") or {}).get("preregister", "tok") for d in Bs}
    declared = pre == {"step"}
    asked = declared or a.decide_on == "step"
    recorded = {(d.get("settings") or {}).get("tps_margin") for d in Bs}
    margin_ok = len(recorded) == 1 and None not in recorded
    if declared and margin_ok:
        margin, msrc = next(iter(recorded)), "pre-registered"
    else:
        margin, msrc = a.tps_margin, "compare option"
    res = {"declared": declared, "post_hoc": a.decide_on == "step" and not declared, "checks": [], "applies": False,
           "margin": margin, "margin_source": msrc}
    # the decision uses the engine's own step time (its decode time per step: boundaries match the step count);
    # the client step time infers the steps inside the client interval from tokens and is shown only
    pred = decode_valid(As[0].get("settings") or {})          # the same predicate as the decode analysis
    valid = [r for d in As + Bs for r in measured(d) if pred(r)]
    server = all(r.get("step_ms_server") for r in valid) and out.get("step_ms_server")
    st = out.get("step_ms_server") if server else out.get("step_ms")
    res["field"] = "step_ms_server" if server else "step_ms"
    if not st:
        res["checks"].append(("step time measured", False, "no step counts"))
        return res
    res["checks"].append(("margin pre-registered (one value in the B runs)", margin_ok,
                          "recorded: %s" % sorted(map(str, recorded))))
    res["checks"].append(("workload valid (no WITHHELD or NOT PUBLISHABLE verdict)",
                          not out["verdict"].startswith(("WITHHELD", "NOT PUBLISHABLE", "NO DECODE")),
                          out["verdict"][:60]))
    res["checks"].append(("arms interleaved in one session", bool(out.get("session_cancels")),
                          "shared and interleaved" if out.get("session_cancels") else
                          "cross-session or not interleaved: the step time carries the session shift too"))
    srcs = {r.get("steps_source") for r in valid}
    res["checks"].append(("steps from engine counters", srcs <= {"metrics", "response"},
                          "sources: %s" % "/".join(sorted(s or "-" for s in srcs))))
    res["checks"].append(("server-side step time in every request", bool(server),
                          "engine decode time per step" if server else
                          "not available: the client step time is an approximation and does not decide"))
    # every decision metric must cover every pair the decode analysis kept: a metric that silently drops pairs
    # (missing telemetry) is an outcome-dependent exclusion of its own
    kept = set(out["decode"]["used_keys"])
    gaps = []
    for nm, r in ((res["field"], st), ("tokens_per_step", out.get("tokens_per_step_ci90")),
                  ("draft_per_step", out.get("draft_per_step_ci90"))):
        if r is None:
            continue
        miss = len(kept - set(r["used_keys"]))
        if miss:
            gaps.append("%s misses %d of %d pairs" % (nm, miss, len(kept)))
    res["checks"].append(("complete coverage (every pair of the decode analysis)", not gaps,
                          "; ".join(gaps) if gaps else "%d pairs" % len(kept)))
    m = math.log(1 + margin / 100)
    for field, label in (("tokens_per_step", "tokens/step"), ("draft_per_step", "drafts/step")):
        c = out.get(field + "_ci90")
        if not c:
            if field == "draft_per_step":
                res["checks"].append(("drafts/step equivalent", None, "not reported by the engine (not checked)"))
                continue
            res["checks"].append(("%s equivalent" % label, False, "not measured"))
            continue
        lo, hi = c["overall"]["ci"]
        res["checks"].append(("%s equivalent within +-%.1f %%" % (label, margin), -m < lo and hi < m,
                              "90 %% CI [%+.2f, %+.2f] %%" % (S.pct(lo), S.pct(hi))))
    lo, hi = st["overall"]["ci"]
    res["verdict"] = ("B FASTER (step time lower)" if hi < 0 else "B SLOWER (step time higher)" if lo > 0
                      else "NO DETECTABLE STEP-TIME CHANGE (MDE80 %.2f %%)" % S.pct(st["mde80"]))
    # the step rule assumes acceptance did not change; a tok/s result significant in the other direction
    # contradicts that assumption (an acceptance loss inside the margin can still outweigh the step gain)
    tlo, thi = out["decode"]["overall"]["ci"]
    # a significant tok/s result can only be confirmed by the step rule, never overturned or neutralised
    conflict = (tlo > 0 and not hi < 0) or (thi < 0 and not lo > 0)
    res["checks"].append(("tok/s verdict not contradicted", not conflict,
                          "tok/s CI [%+.2f, %+.2f] %%" % (S.pct(tlo), S.pct(thi))))
    # the point estimate of tok/s is the step effect plus the tokens/step change (identity per request): if the
    # measured acceptance change outweighs the step gain, the step verdict does not describe the lever
    test = out["decode"]["overall"]["est"]
    agree = (hi < 0 and test > 0) or (lo > 0 and test < 0) or not (hi < 0 or lo > 0)
    res["checks"].append(("tok/s point estimate in the step verdict's direction", agree,
                          "tok/s %+.2f %% = step %+.2f %% + tokens/step %+.2f %%" % (
                              S.pct(test), S.pct(-st["overall"]["est"]),
                              S.pct(out["tokens_per_step"]["overall"]["est"]) if out.get("tokens_per_step") else 0)))
    if As[0]["settings"].get("mode") in SCREEN_MODES:
        res["verdict"] = "SCREEN ONLY (%s mode): " % As[0]["settings"]["mode"] + res["verdict"]
    res["implied_tok_s"] = [-st["overall"]["est"], -hi, -lo]
    tc = out.get("tokens_per_step_ci90")
    if tc:      # what tok/s can be, given the step-time CI and the tokens/step 90 % CI together
        res["tok_s_bound"] = [-hi + tc["overall"]["ci"][0], -lo + tc["overall"]["ci"][1]]
    res["passed"] = all(ok is not False for _, ok, _ in res["checks"])
    res["applies"] = declared and res["passed"]
    res["asked"] = asked
    return res


def guard_sessions(As, Bs):
    """quality guard per session: runs paired by position (session i of A with session i of B); every pair is
    compared, and runs without a partner are listed with their pass counts"""
    out = []
    for i in range(max(len(As), len(Bs))):
        x = As[i] if i < len(As) else None
        y = Bs[i] if i < len(Bs) else None
        gx, gy = (x or {}).get("guard"), (y or {}).get("guard")
        if gx and gy:
            out.append(dict(G.compare_guard(gx, gy), A=name(x), B=name(y)))
        elif gx or gy:
            g, d = (gx, x) if gx else (gy, y)
            out.append({"single": name(d), "passed": g["passed"], "total": g["total"]})
    return out or None


def lower_is_better(res):
    lo, hi = res["overall"]["ci"]
    if hi < 0:
        return "B FASTER (lower)"
    if lo > 0:
        return "B SLOWER (higher)"
    return "NO DETECTABLE DIFFERENCE (MDE80 %.1f %%)" % S.pct(res["mde80"])


def seed_analysis(As, Bs, res, cluster, valid=None):
    """T>0: split the per-prompt variance of the paired log ratio into a between-prompt part and a seed (sampling)
    part, and project the MDE for more seeds. Pairs are per (prompt, seed); the verdict's CI is over prompts, with
    each prompt's seeds averaged, so it already contains the sampling variance."""
    get = (lambda d: logs(d, lambda r: r.get("tok_s"), valid) if valid else logs(d, lambda r: r.get("tok_s")))
    ra, rb = by_boot(As, get), by_boot(Bs, get)
    keys = [k for k in set(ra[0]).intersection(*ra[1:], *rb) if cluster.get(k) in res["per_prompt"]]
    d = {k: statistics.fmean(r[k] for r in rb) - statistics.fmean(r[k] for r in ra) for k in keys}
    groups = {}
    for k in keys:
        groups.setdefault(cluster[k], []).append(d[k])
    multi = [v for v in groups.values() if len(v) >= 2]
    if not multi:
        return None
    within = sum(sum((x - statistics.fmean(v)) ** 2 for x in v) for v in multi) / sum(len(v) - 1 for v in multi)
    n_seeds = statistics.median(len(v) for v in groups.values())
    # share of the observed per-prompt variance that is seed (sampling) variance; the rest is between prompts
    share = min(1.0, within / n_seeds / res["sd_d"] ** 2) if res["sd_d"] > 0 else 1.0
    o = res["overall"]
    vb = o["se"] ** 2 - o["se_prompt"] ** 2                  # session/boot part of the variance
    zsum = S.t_ppf(0.975, o["df"]) + S.t_ppf(0.8, o["df"])
    proj = {}
    for m in (1, 2, 4):
        sv = n_seeds * m
        proj[int(sv)] = zsum * math.sqrt(o["se_prompt"] ** 2 * ((1 - share) + share * n_seeds / sv) + vb)
    return {"seeds": n_seeds, "sd_pair": math.sqrt(within), "seed_share": share,
            "sd_between": res["sd_d"] * math.sqrt(1 - share), "mde_by_seeds": proj,
            "prompts_for_1pct": res["prompts_for_1pct"].get(1), "n_prompts": res["n_prompts"]}


def _try(f):
    try:
        return f()
    except ValueError:
        return None


def report(out, As, Bs, problems, notes, a):
    w = sys.stdout.write
    d = out["decode"]
    o = d["overall"] if d else None
    w("speedbench compare  A=%s  B=%s\n" % (" + ".join(out["A"]), " + ".join(out["B"])))
    for p in problems:
        w("WARNING (forced): %s\n" % p)
    for m in out.get("session_warnings") or []:
        w("\n!!! %s\n" % m)
    if out.get("session_warnings"):
        w("\n")
    for n in notes:
        w("note: %s\n" % n)
    s0 = As[0]["settings"]
    w("prompt set %s (%s), mode %s, max_tokens %s, T=%s, effort %s, conc %s, prime %s, token boundary %s\n" % (
        As[0]["prompt_set"]["version"], As[0]["prompt_set"]["sha256"][:12], s0["mode"], s0["max_tokens"],
        s0["temperature"], s0["effort"], s0["conc"], s0["prime"], out["boundary"]))
    if d:
        report_decode(out, As, Bs, a, w)
    else:
        w("\nDECODE tok/s: %s\n" % out["verdict"])
    ib = out["identical_between_arms"]
    unit = "(prompt, seed) pairs" if s0_temp(As) else "prompts"
    w("\nOUTPUT IDENTITY (all runs, repeats and variants): %d/%d %s identical in A and B\n" % (ib[0], ib[1], unit))
    if out.get("agent") and out["agent"].get("isolate") == "text":
        w("  agent mode with --isolate text: the salt line differs between the arms, so outputs differ by design\n"
          "  (use --isolate field on engines that honour cache_salt to compare outputs)\n")
    elif s0_temp(As) and ib[1] and ib[0] < ib[1]:
        w("  T>0: identical only where the engine reproduces a seed and the lever leaves the random draws alone\n")
    elif ib[1] and ib[0] == ib[1]:
        w("  all identical: free of trajectory luck (remaining noise: timing, boots); same tokens and steps in\n"
          "  both arms, so the tok/s verdict IS the step-time verdict\n")
    elif ib[1] and ib[0] >= 0.9 * ib[1]:
        w("  mostly identical: little trajectory luck; check the differing prompts in the per-prompt list\n")
    elif ib[1]:
        w("  outputs differ: per-prompt acceptance luck is part of the noise (expect per-prompt SD of ~4-6 %)\n")
    for nm in ("A", "B"):
        x = out.get("identical_within_" + nm)
        if x:
            w("  within %s: %d/%d prompts identical across runs/repeats%s\n" % (
                nm, x[0], x[1], "" if x[0] == x[1] else " (T=0 nondeterminism: repeats add trajectories)"))
    dv = out.get("divergence")
    if dv and dv["differing"] and dv.get("positions"):
        e = dv["early"]
        w("  DIVERGENCE of the differing outputs (first differing token, approx.; median output %d tokens):\n"
          "    n=%d  min %d  p25 %.0f  median %.0f  p75 %.0f; diverging before token 16: %d/%d, before 64: %d/%d\n" % (
              dv["median_output_tokens"], len(dv["positions"]), dv["min"], dv["p25"], dv["median"], dv["p75"],
              e[16], len(dv["positions"]), e[64], len(dv["positions"])))
        if e[16] > 0.5 * len(dv["positions"]):
            w("    most outputs differ from the first tokens: this looks like a change of the prompt, chat template,\n"
              "    sampling or a broken kernel rather than a floating-point-order change; check the outputs\n")
        else:
            w("    late divergence is typical of numerics changes (a near-tie flips somewhere in the output)\n")
    elif dv is None and not s0_temp(As) and ib[1] and ib[0] < ib[1] and not (
            out.get("agent") and out["agent"].get("isolate") == "text"):
        w("  (divergence summary needs the generated text: run with --quality-guard or --keep-text)\n")
    gl = out.get("guard")
    if gl:
        w("\nQUALITY GUARD (exact-answer items at T=0, per session; a guard against gross breakage, not a quality "
          "benchmark)\n")
        for gd in gl:
            if gd.get("single"):
                w("  %s: %d/%d passed (no run of the other arm in this session position: not compared)\n" % (
                    gd["single"], gd["passed"], gd["total"]))
                continue
            w("  %s vs %s: A %d/%d, B %d/%d; lost in B: %s; gained in B: %s; identical outputs %d/%d\n" % (
                gd["A"], gd["B"], gd["passed_A"], gd["n"], gd["passed_B"], gd["n"], ", ".join(gd["lost_in_B"]) or "none",
                ", ".join(gd["won_in_B"]) or "none", gd["identical_outputs"], gd["n"]))
            if gd["truncated_A"] or gd["truncated_B"]:
                w("    no answer within max_tokens: A %d, B %d (raise --guard-max-tokens)\n" % (
                    gd["truncated_A"], gd["truncated_B"]))
            w("    -> %s (one-sided exact McNemar p=%.3g)\n" % (
                "FLAG: B LOST ITEMS - check the outputs before trusting the speed result" if gd["drop"] else
                "no drop detected", gd["p_one_sided"]))
    if d:
        report_decomposition(out, w)
    sr = out.get("step_rule") or {}
    if sr.get("verdict"):
        w("\nSTEP-TIME RULE (pre-registered with run --preregister step; %s; margin %.1f %% from %s; on %s)\n" % (
            "declared" if sr["declared"] else "post hoc: shown, does not decide" if sr["post_hoc"] else "not declared",
            sr["margin"], sr["margin_source"], "server step time" if sr["field"] == "step_ms_server" else
            "client step time (approximate)"))
        for nm, ok, why in sr["checks"]:
            w("  [%s] %s: %s\n" % ("ok" if ok else "--" if ok is None else "FAIL", nm, why))
        lo_i, hi_i = sr["implied_tok_s"][1], sr["implied_tok_s"][2]
        w("  step-time verdict: %s; implied tok/s effect at equal acceptance %+.2f %% [%+.2f, %+.2f]\n" % (
            sr["verdict"], S.pct(sr["implied_tok_s"][0]), S.pct(lo_i), S.pct(hi_i)))
        if sr.get("tok_s_bound"):
            w("  tok/s effect allowing for the measured acceptance uncertainty: [%+.2f, %+.2f] %%\n" % (
                S.pct(sr["tok_s_bound"][0]), S.pct(sr["tok_s_bound"][1])))
        if sr["applies"]:
            w("  -> DECISION: the step-time verdict (all checks passed, declared before the run)\n")
        else:
            w("  -> DECISION: the tok/s verdict above (%s)\n" % (
                "step rule not declared before the run" if not sr["declared"] else "a check failed"))
    for k, nm in (("ttft", "TTFT measured request"), ("ttft_prime", "TTFT priming request")):
        r = out.get(k)
        if r:
            w("%-24s B vs A %s  (lower is better)\n" % (nm, fmt_ci(r)))
    ag = out.get("agent")
    if ag:
        w("\nAGENT (multi-turn, prefix cache; pairs per turn, CI clustered by conversation; isolation %s; "
          "lower is better)\n" % ag["isolate"])
        for k, nm in (("ttft_cold", "TTFT turn 1 (cold)"), ("ttft_cached", "TTFT turns 2+ (cached)"),
                      ("e2e", "turn latency (end to end)")):
            r = ag.get(k)
            if r:
                w("  %-26s B vs A %s  %d conversations, df %.0f, MDE80 %.2f %%   -> %s\n" % (
                    nm, fmt_ci(r), r["n_prompts"], min(r["overall"]["df"], 9999), S.pct(r["mde80"]), ag[k + "_verdict"]))
            elif ag.get(k + "_verdict"):
                w("  %-26s %s\n" % (nm, ag[k + "_verdict"]))
        if ag.get("per_turn"):
            w("  per turn index, B vs A [95 %% CI over conversations] (descriptive):\n  %-5s %-30s %s\n" % (
                "turn", "TTFT", "decode tok/s"))
            for t, v in ag["per_turn"].items():
                w("  t%-4d %-30s %s\n" % (t, fmt_ci(v["ttft"]) if v["ttft"] else "-",
                                           fmt_ci(v["tok_s"]) if v["tok_s"] else "-"))
        for nm, sm in ag["summaries"].items():
            c, h = sm.get("cached") or {}, sm.get("cold") or {}
            w("  %s: cold TTFT %s ms (%s prompt tokens); cached turns: TTFT %s ms, %s new of %s prompt tokens, "
              "cache hit vs ideal %s, decode %s tok/s, e2e %s s\n" % (
                  nm, h.get("median_ttft_ms"), h.get("median_prompt_tokens"), c.get("median_ttft_ms"),
                  c.get("median_new_tokens", "?"), c.get("median_prompt_tokens"), c.get("cache_hit_vs_ideal", "n/a"),
                  c.get("median_tok_s"), c.get("median_e2e_s")))
            if sm.get("isolation"):
                w("    cache isolation: %s\n" % sm["isolation"])
            if sm.get("warning"):
                w("  WARNING %s: %s\n" % (nm, sm["warning"]))
        w("  history is frozen (recorded assistant replies), so TTFT is free of trajectory luck; decode tok/s is not\n")
    en = out.get("energy")
    em = out.get("energy_meta")
    if em:
        w("\nENERGY (%s; %s)\n" % ("/".join(sorted(map(str, em["kind"]))), em["note"]))
        if len(em["kind"]) > 1:
            w("  WARNING: the arms used different kinds of power source: not comparable\n")
        if en:
            w("  decode J/token B vs A %s  (lower is better)  MDE80 %.2f %%\n" % (fmt_ci(en), S.pct(en["mde80"])))
        for nm, e in em["per_arm"].items():
            if e:
                w("  %s: %.4f J/token decode (%.2f tok/J), %.0f W mean while decoding, idle %s W%s\n" % (
                    nm, e["decode_j_per_tok"], e["decode_tok_per_j"], e.get("decode_mean_w") or 0, em["idle_w"].get(nm),
                    ", %.4f J/token above idle" % e["decode_j_per_tok_above_idle"]
                    if e.get("decode_j_per_tok_above_idle") is not None else ""))
    if out["horizons"]:
        w("\nHORIZON (per-prompt SD of the log ratio over the first h tokens):\n")
        for h, v in sorted(out["horizons"].items()):
            w("  h=%4d  n=%d  SD %.2f %%  estimate %+.2f %%\n" % (h, v["n"], v["sd_d"] * 100, S.pct(v["est"])))
        if d:
            w("  h=full  n=%d  SD %.2f %%  estimate %+.2f %%\n" % (d["n_prompts"], d["sd_d"] * 100, S.pct(o["est"])))
    for nm, info in out["runs"].items():
        s = info["summary"]
        line = "run %s: %d/%d requests ok, %d valid decode, %d exact boundary" % (
            nm, s["requests"] - s["failed"], s["requests"], s["valid_decode"], s.get("exact_boundary", 0))
        sen = s.get("sentinel")
        if sen:
            line += "; sentinels %d/%d identical, drift %s %%" % (sen["identical_text"], sen["n"], sen["drift_pct_all"])
        w(line + "\n")
        for e in s.get("errors") or []:
            w("    error: %s\n" % e)


def report_decode(out, As, Bs, a, w):
    d = out["decode"]
    o = d["overall"]
    s0 = As[0]["settings"]
    w("\nDECODE tok/s, B vs A (paired over %d %s; %d excluded%s%s)\n" % (
        d["n_prompts"], out["unit"], len(d["excluded"]), (": " + ", ".join(d["excluded"][:8])) if d["excluded"] else "",
        "; %d of %d %s dropped" % (d["pairs_dropped"], d["pairs_planned"], "(prompt, seed) pairs" if s0_temp(As)
                                    else "turns") if out.get("partial") else ""))
    w("  overall   %s  p=%.3g  df=%.0f   -> %s%s\n" % (
        fmt_ci(d), o["p"], min(o["df"], 9999), out["verdict"],
        "  [interleaved in one session: boot shocks cancel]" if out.get("session_cancels") else
        "  [prior-conditional: boot SD assumed]" if out["prior_conditional"] else ""))
    for bw in out["boot_warnings"]:
        w("  WARNING: %s\n" % bw)
    ex = out.get("exclusions")
    if ex and ex["dropped"]:
        w("  exclusions: %d dropped (missing in A only %d, B only %d, both %d; balance p=%.2g)%s\n" % (
            ex["dropped"], ex["a_only"], ex["b_only"], ex["both"], ex["p_imbalance"],
            "; worst-case imputation at the least favourable observed difference: %+.2f %% [%+.2f, %+.2f]" % (
                S.pct(ex["bound_est"]), S.pct(ex["bound_ci"][0]), S.pct(ex["bound_ci"][1]))
            if ex.get("bound_ci") else ""))
    if out.get("decode_data_only"):
        w("  session term from the interleaved sessions alone (no prior): CI %s\n" % (
            fmt_ci(out["decode_data_only"]).split("  ", 1)[1]))
    if out.get("decode_boot05"):
        w("  if the session SD were %.1f %% instead of %.2f %%: CI %s\n" % (
            a.alt_prior, a.boot_prior, fmt_ci(out["decode_boot05"]).split("  ", 1)[1]))
    nd = d["prompts_for_1pct"]
    w("  per-prompt SD of log ratio %.2f %%   MDE(80 %% power) %.2f %%\n" % (d["sd_d"] * 100, S.pct(d["mde80"])))
    w("  prompts needed for a 1 %% MDE with 1/2/4 boots per arm: %s\n" % " / ".join(
        str(nd[R]) if nd[R] else "unreachable" for R in (1, 2, 4)))
    bt = d["boot"]
    w("  boot term: A sigma %.2f %% (%s, %d boot%s), B sigma %.2f %% (%s, %d boot%s)\n" % (
        bt["A"]["sigma_b"] * 100, bt["A"]["source"], bt["A"]["runs"], "s" * (bt["A"]["runs"] > 1),
        bt["B"]["sigma_b"] * 100, bt["B"]["source"], bt["B"]["runs"], "s" * (bt["B"]["runs"] > 1)))
    if bt["A"].get("sigma_session_anova") is not None:
        w("  session-to-session SD of the paired difference (from %d interleaved sessions): %.2f %% (ANOVA, %d df), "
          "%.2f %% pooled with the %.2f %% prior\n" % (bt["A"]["runs"], bt["A"]["sigma_session_anova"] * 100,
                                                     bt["A"]["anova_df"], bt["A"]["sigma_pooled"] * 100, a.shared_prior))
    for x in ("A", "B"):
        if bt[x].get("sigma_b_anova") is not None:
            w("  arm %s: session SD estimated from %d sessions %.2f %% (ANOVA, %d df)\n" % (
                x, bt[x]["runs"], bt[x]["sigma_b_anova"] * 100, bt[x]["anova_df"]))
    w("  sensitivity (descriptive): median %+.2f %%, Hodges-Lehmann %+.2f %%, 3-MAD winsorized %+.2f %%%s, "
      "bootstrap CI (prompts only) [%+.2f, %+.2f], max leave-one-out shift %.2f pp\n" % (
          S.pct(d["median_d"]), S.pct(d["hodges_lehmann"]), S.pct(d["winsorized_mean"]),
          (" (clipped: %s)" % ", ".join(d["winsorized"])) if d["winsorized"] else "",
          S.pct(d["bootstrap_ci_prompt_only"][0]), S.pct(d["bootstrap_ci_prompt_only"][1]), d["loo_max_shift"] * 100))
    w("  largest losses: %s\n" % ", ".join("%s %+.1f %%" % (k, S.pct(v)) for k, v in d["worst"]))
    w("  largest gains:  %s\n" % ", ".join("%s %+.1f %%" % (k, S.pct(v)) for k, v in d["best"]))
    sm = out.get("sampling")
    if s0_temp(As):
        w("\n  T>0 (T=%s, seeds %s): pairs per (prompt, seed), averaged per prompt; the CI is over prompts and includes\n"
          "  the sampling variance. At T>0 the text depends on the seed AND on everything that changes the random\n"
          "  draws (drafting, batching), so every lever is trajectory-changing: more prompts or seeds are needed.\n" % (
              As[0]["settings"].get("temperature"), As[0]["settings"].get("seeds")))
        if sm:
            w("  per-(prompt, seed) SD of the paired log ratio %.2f %%; %s seeds per prompt; seed sampling = %.0f %% of the\n"
              "  per-prompt variance (between-prompt SD %.2f %%)\n" % (
                sm["sd_pair"] * 100, _g(sm["seeds"]), sm["seed_share"] * 100, sm["sd_between"] * 100))
            w("  MDE80 with %d prompts: %s; prompts for a 1 %% MDE at %s seeds: %s\n" % (
                sm["n_prompts"], ", ".join("%d seeds %.2f %%" % (k, S.pct(v)) for k, v in sorted(sm["mde_by_seeds"].items())),
                _g(sm["seeds"]), sm["prompts_for_1pct"] or "unreachable (session term)"))
        else:
            w("  (seed variance not estimable: one seed per prompt)\n")
    sa = As[0]["summary"]
    sb = Bs[0]["summary"]
    w("\n  per category (descriptive, no verdicts; medians and reasoning share from the first run of each arm):\n")
    w("  %-13s %4s  %-26s %9s %9s   %s\n" % ("category", "n", "B vs A [95 % CI]", "A med", "B med", "reasoning share A/B"))

    def rshare(s):
        return s.get("reasoning_token_frac", s.get("reasoning_char_frac", "-"))
    for c, r in d["categories"].items():
        ca, cb = sa.get(c) or {}, sb.get(c) or {}
        w("  %-13s %4d  %-26s %9s %9s   %s/%s\n" % (c, r["n"], fmt_ci(r), ca.get("median_tok_s", "-"),
                                                  cb.get("median_tok_s", "-"), rshare(ca), rshare(cb)))
    ca, cb = sa.get("ALL") or {}, sb.get("ALL") or {}
    w("  %-13s %4s  %-26s %9s %9s   %s/%s\n" % ("ALL", "", "", ca.get("median_tok_s", "-"), cb.get("median_tok_s", "-"),
                                              rshare(ca), rshare(cb)))
    if ca.get("aggregate_tok_s") and cb.get("aggregate_tok_s"):
        w("  token-weighted aggregate (decset-style): A %.2f  B %.2f  (%+.2f %%)\n" % (
            ca["aggregate_tok_s"], cb["aggregate_tok_s"], (cb["aggregate_tok_s"] / ca["aggregate_tok_s"] - 1) * 100))


def report_decomposition(out, w):
    w("\nDECOMPOSITION (descriptive; steps from %s)\n" % (out["steps_source"] or "-"))
    if out["steps_source"] == "stream":
        w("  steps counted from stream events: valid only if the server emits one event per engine step\n"
          "  (gr8 drops empty events; use --metrics on servers with vLLM spec-decode counters)\n")
    if out.get("steps_agreement"):
        g = out["steps_agreement"]
        w("  stream events vs engine step counters agree (+-2) on %d/%d requests\n" % g)
    for field, nm in DECOMP:
        r = out.get(field)
        if r:
            w("  %-20s B vs A %s   per-prompt SD %.2f %%   MDE80 %.2f %%\n" % (
                nm, fmt_ci(r), r["sd_d"] * 100, S.pct(r["mde80"])))
    present = [(f, nm) for f, nm in DECOMP if out.get(f)]
    if present:
        w("  per category, B vs A [95 %% CI] (descriptive):\n  %-13s %s\n" % (
            "category", "".join("%-28s" % nm for _, nm in present)))
        for c in out["decode"]["categories"]:
            cells = []
            for f, _ in present:
                r = out[f]["categories"].get(c)
                cells.append("%-28s" % (fmt_ci(r) if r else "-"))
            w("  %-13s %s\n" % (c, "".join(cells)))
    w("  identity: log tok/s = log tokens/step - log step time (per request, exactly)\n")


def s0_temp(As):
    return bool(As[0]["settings"].get("temperature"))


def _g(x):
    return ("%d" % x) if float(x).is_integer() else ("%.1f" % x)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="speedbench compare", description=__doc__)
    ap.add_argument("files", nargs="*", help="A.json B.json (one run per arm)")
    ap.add_argument("--a", nargs="+", default=[], help="run files of arm A")
    ap.add_argument("--b", nargs="+", default=[], help="run files of arm B")
    ap.add_argument("--boot-prior", type=float, default=1.0,
                    help="session SD in %% of one arm's whole run, assumed when an arm has one session and pooled "
                         "with the estimate otherwise (default 1.0: same config ~2 %% apart across sessions on our "
                         "fleet; 0.25 was the within-session value)")
    ap.add_argument("--alt-prior", type=float, default=2.0,
                    help="second session SD in %% shown for a prior-conditional CI (default 2.0)")
    ap.add_argument("--shared-prior", type=float, default=0.1,
                    help="residual SD in %% of the paired difference for arms interleaved in one session "
                         "(default 0.1; pooled with the estimate when there are >= 2 sessions)")
    ap.add_argument("--decide-on", choices=["tok", "step"], default="tok",
                    help="show the step-time rule even if the run did not pre-register it (post hoc, never decides)")
    ap.add_argument("--tps-margin", type=float, default=2.0,
                    help="equivalence margin in %% for tokens/step and drafts/step in the step-time rule, used only "
                         "when the run did not pre-register one (run --tps-margin; default 2)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--force", action="store_true", help="compare runs that fail the compatibility checks")
    ap.add_argument("--json", help="write the full analysis as JSON")
    a = ap.parse_args(argv)
    fa, fb = a.a, a.b
    if a.files:
        if fa or fb or len(a.files) != 2:
            raise SystemExit("give either two positional files or --a ... --b ...")
        fa, fb = [a.files[0]], [a.files[1]]
    if not fa or not fb:
        raise SystemExit("need at least one run per arm")
    As, Bs = [load(p) for p in fa], [load(p) for p in fb]
    problems, notes, boundary = check_inputs(As, Bs, a.force)
    out = analyse(As, Bs, a, boundary)
    report(out, As, Bs, problems, notes, a)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
