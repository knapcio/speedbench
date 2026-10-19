"""Regression tests for the v0.3 review: each test reproduces a reviewer scenario."""
import argparse
import contextlib
import io
import json
import math
import os
import random
import tempfile
import unittest

from speedbench import agent, compare, guard, power, report, run, selftest
from tests.test_e2e_mock import Mock

ARGS = argparse.Namespace(boot_prior=1.0, shared_prior=0.1, alt_prior=2.0, alpha=0.05, decide_on="tok", tps_margin=2.0)


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
        res = f(*a, **k)
    return res, out.getvalue()


def agent_docs(conv_effects, turns=6, turn_sd=0.002, mode="agent", decode_tokens=150, seed=1):
    """synthetic interleaved agent runs: B's TTFT and latency change by a per-conversation effect"""
    rng = random.Random(seed)
    docs = []
    for arm in ("A", "B"):
        reqs = []
        for c, u in enumerate(conv_effects):
            for t in range(1, turns + 1):
                f = math.exp((u if arm == "B" else 0.0) + rng.gauss(0, turn_sd))
                key = "conv-%02d@t%d" % (c + 1, t)
                reqs.append({"key": key, "id": key, "conv": "conv-%02d" % (c + 1), "turn": t, "category": "t%d" % t,
                             "kind": "measure", "ok": True, "valid_decode": decode_tokens >= 32,
                             "decode_tokens": decode_tokens, "tok_s": 40 * math.exp(rng.gauss(0, 0.03)),
                             "ttft_s": (2.0 if t == 1 else 0.2) * f, "total_s": 4.0 * f, "sha256": arm + key,
                             "repeat": 0, "seed": None, "boundary": "exact", "t_wall": 2 * len(reqs) + (arm == "B")})
        docs.append({"label": arm, "run_id": arm, "session_id": "s", "prompt_set": {"version": "v", "sha256": "x" * 64},
                     "settings": {"mode": mode, "temperature": 0.0, "max_tokens": 160, "effort": "low", "conc": 1,
                                  "prime": False, "isolate": "text"},
                     "summary": {"requests": len(reqs), "failed": 0, "valid_decode": len(reqs),
                                 "agent": {"isolation": "verified: turn 1 missed the cache in every conversation"}},
                     "requests": reqs})
    return docs


class AgentClustering(unittest.TestCase):
    """#1: turns of one conversation are correlated; the CI must be over conversations"""

    def test_conversation_effects_do_not_become_significant(self):
        effects = [0.03, 0.02, 0.04, -0.01, 0.01, 0.02, -0.02, 0.03]     # mean +1.5 %, SD 2.1 % across conversations
        a, b = agent_docs(effects)
        out = compare.analyse([a], [b], ARGS)
        r = out["agent"]["ttft_cached"]
        self.assertEqual(r["n_prompts"], 8)                               # clusters = conversations
        self.assertEqual(r["pairs_planned"], 40)
        self.assertLessEqual(r["overall"]["ci"][0], 0)                    # per-turn units would call this significant
        self.assertIn("NO DETECTABLE DIFFERENCE", out["agent"]["ttft_cached_verdict"])
        self.assertLessEqual(r["overall"]["df"], 7.5)
        _, txt = quiet(compare.report, out, [a], [b], [], [], ARGS)
        self.assertIn("CI clustered by conversation", txt)

    def test_agent_quick_latency_is_screen_only(self):                  # 9
        a, b = agent_docs([0.05] * 4, turns=4, mode="agent-quick")
        out = compare.analyse([a], [b], ARGS)
        for k in ("ttft_cached_verdict", "e2e_verdict", "ttft_cold_verdict"):
            self.assertTrue(out["agent"][k].startswith("SCREEN ONLY (agent-quick mode): B SLOWER"), out["agent"][k])


class AgentLatencyValidity(unittest.TestCase):
    """#4: short tool-call outputs have valid timings; they must not suppress latency or abort compare"""

    def test_short_outputs(self):
        a, b = agent_docs([0.05] * 8, decode_tokens=4)                    # every output too short to time decode
        out = compare.analyse([a], [b], ARGS)                             # used to raise ValueError
        self.assertIsNone(out["decode"])
        self.assertTrue(out["verdict"].startswith("NO DECODE VERDICT"))
        self.assertEqual(out["agent"]["ttft_cached_verdict"], "B SLOWER (higher)")
        self.assertEqual(out["agent"]["e2e_verdict"], "B SLOWER (higher)")
        _, txt = quiet(compare.report, out, [a], [b], [], [], ARGS)
        self.assertIn("NO DECODE VERDICT", txt)
        self.assertIn("AGENT", txt)

    def test_tool_call_turns_are_timed_for_decode(self):
        a, b = agent_docs([0.0] * 8, decode_tokens=20)                    # >= 8 tokens: timed in agent mode
        out = compare.analyse([a], [b], ARGS)
        self.assertEqual(out["decode"]["pairs_dropped"], 0)


class OutcomeDependentExclusion(unittest.TestCase):
    """#2: the reviewer's 40 prompts x 10 seeds case: one slow, short seed per prompt, complete-data effect zero"""

    def test_b_only_untimeable(self):
        a, b = selftest.trap_docs(random.Random(1), mode="b_only", short_tokens=4)
        out = compare.analyse([a], [b], ARGS)
        self.assertFalse(out["verdict"].endswith("B FASTER"), out["verdict"])
        self.assertTrue(out["verdict"].startswith("WITHHELD"))
        self.assertEqual(out["exclusions"]["b_only"], 40)

    def test_short_but_timeable_seeds_are_kept(self):
        a, b = selftest.trap_docs(random.Random(2), mode="b_only", short_tokens=20)
        out = compare.analyse([a], [b], ARGS)
        self.assertEqual(out["decode"]["pairs_dropped"], 0)
        self.assertNotIn("B FASTER", out["verdict"])

    def test_untimeable_in_both_arms(self):
        a, b = selftest.trap_docs(random.Random(3), mode="both", short_tokens=4)
        out = compare.analyse([a], [b], ARGS)
        self.assertTrue(out["verdict"].startswith("WITHHELD"), out["verdict"])

    def test_imbalance_below_the_cap(self):
        # 5 of 400 pairs dropped (under the 5 % cap), all in B, all B's slow seeds: attribution withholds
        a, b = selftest.trap_docs(random.Random(4), mode="b_only", short_tokens=20)
        for r in b["requests"]:
            if r["seed"] == int(r["id"][1:]) % 10 and int(r["id"][1:]) < 5:
                r["decode_tokens"] = 4
        out = compare.analyse([a], [b], ARGS)
        self.assertEqual(out["exclusions"]["b_only"], 5)
        self.assertTrue(out["verdict"].startswith("WITHHELD"), out["verdict"])

    def test_selftest_trap(self):
        for row in selftest.exclusion_trap(sims=5):
            self.assertEqual(row["b_faster"], 0.0, row)


class EnergyGaps(unittest.TestCase):
    def test_request_inside_a_long_gap(self):                            # 5
        self.assertIsNone(power.integrate([0, 100], [100, 200], 50, 50.2, 1))
        self.assertIsNotNone(power.integrate([0, 0.5, 1.0], [100, 100, 100], 0.6, 0.7, 1))

    def test_load_report_states_the_scope(self):                         # 10
        doc = {"label": "l", "settings": {"conc": [1]}, "levels": [],
               "energy": {"note": power.NOTE["gpu"], "kind": "gpu", "idle_w": 12.0, "sources": []}}
        buf = io.StringIO()
        report.print_load(doc, buf.write)
        self.assertIn("excludes CPU", buf.getvalue())


class CacheIsolationEvidence(unittest.TestCase):
    def test_unverified_without_cached_counts(self):                     # 6
        m = Mock("--step-ms", "1", "--no-cache-report", "--ignore-cache-salt")
        try:
            t = tempfile.mkdtemp()
            paths, _ = quiet(run.main, ["--arm", "A=" + m.url, "--arm", "B=" + m.url, "--mode", "agent-quick",
                                        "--isolate", "field", "--out", os.path.join(t, "u-"), "--quiet",
                                        "--warmup", "1", "--warmup-prefill", "0"])
            with open(paths[1], encoding="utf-8") as f:
                b = json.load(f)
            self.assertTrue(b["summary"]["agent"]["isolation"].startswith("UNVERIFIED"))
            self.assertIn("--enable-prompt-tokens-details", b["summary"]["agent"]["isolation"])
            _, txt = quiet(compare.main, paths)
            self.assertIn("cache isolation: UNVERIFIED", txt)
        finally:
            m.stop()

    def test_verified_and_failed_states(self):
        rows = [{"kind": "measure", "ok": True, "conv": "c", "turn": 1, "repeat": 0, "seed": None, "prompt_tokens": 100,
                 "cached_tokens": c, "ttft_s": 0.1, "total_s": 1.0} for c in (0,)]
        self.assertTrue(agent.summarize(rows)["isolation"].startswith("verified"))
        rows[0]["cached_tokens"] = 90
        self.assertTrue(agent.summarize(rows)["isolation"].startswith("FAILED"))


class GuardPerSession(unittest.TestCase):
    def test_second_session_failure_is_flagged(self):                   # 7
        def g(n_pass):
            items = [{"id": "i%d" % k, "pass": k < n_pass, "sha256": "h%d%d" % (k, n_pass)} for k in range(30)]
            return {"items": items, "passed": n_pass, "total": 30, "truncated": 0}
        A = [{"label": "A", "run_id": "a%d" % i, "guard": g(30)} for i in range(2)]
        B = [{"label": "B", "run_id": "b0", "guard": g(30)}, {"label": "B", "run_id": "b1", "guard": g(0)}]
        res = compare.guard_sessions(A, B)
        self.assertEqual(len(res), 2)
        self.assertFalse(res[0]["drop"])
        self.assertTrue(res[1]["drop"])


class DivergenceToolNames(unittest.TestCase):
    def test_changed_function_name(self):                                # 8
        a = {"text": {"reasoning": "", "content": "", "tool_calls": [{"name": "read_file", "arguments": "{}"}]},
             "completion_tokens": 5}
        b = {"text": {"reasoning": "", "content": "", "tool_calls": [{"name": "run_shell", "arguments": "{}"}]}}
        self.assertEqual(guard.first_divergence(a, b), 0)


if __name__ == "__main__":
    unittest.main()


class VerificationRound(unittest.TestCase):
    """the verification round of the v0.3 review"""

    def test_drops_below_the_cap_are_not_publishable(self):
        # 40 prompts x 25 seeds, one slow seed per prompt untimeable in both arms: 4 % dropped, complete effect zero
        a, b = selftest.trap_docs(random.Random(5), mode="both", short_tokens=4, seeds=25)
        out = compare.analyse([a], [b], ARGS)
        self.assertEqual(out["decode"]["pairs_dropped"], 40)
        self.assertTrue(out["verdict"].startswith("NOT PUBLISHABLE (40 of 1000 pairs dropped"), out["verdict"])
        _, txt = quiet(compare.report, out, [a], [b], [], [], ARGS)
        self.assertIn("NOT PUBLISHABLE", txt)
        self.assertIn("exclusions: 40 dropped", txt)

    def test_any_drop_blocks_publication_in_every_mode(self):
        a, b = agent_docs([0.05] * 8)                                     # agent mode, T=0
        for r in b["requests"]:
            if r["key"] == "conv-01@t3":
                r["ok"] = False
        out = compare.analyse([a], [b], ARGS)
        self.assertTrue(out["agent"]["e2e_verdict"].startswith("NOT PUBLISHABLE (1 of 48 pairs dropped"))
        self.assertTrue(out["verdict"].startswith("NOT PUBLISHABLE"))

    def test_sensitivity_must_keep_the_direction(self):
        from speedbench import stats as S
        chk = {"a_only": 0, "b_only": 0, "p_imbalance": 1.0, "dropped": 3, "direction": 1,
               "bound_ci": [math.log(1 - 0.0863), math.log(1 - 0.0827)]}       # reviewer: -8.45 % [-8.63, -8.27]
        why = S.exclusion_rule(chk, 100)
        self.assertIsNotNone(why)
        self.assertIn("reverses the verdict", why)
        chk["bound_ci"] = [0.001, 0.02]
        self.assertIsNone(S.exclusion_rule(chk, 100))
        chk.update(direction=-1, bound_ci=[0.001, 0.02])                  # a B SLOWER verdict reversed
        self.assertIsNotNone(S.exclusion_rule(chk, 100))

    def test_boundary_check_uses_the_analysis_threshold(self):
        # 25 completion tokens in 1 s: exact counting gives A 20 tokens, the fallback gives B 24 (+20 %)
        t = tempfile.mkdtemp()
        paths = []
        for arm, exact in (("A", True), ("B", False)):
            reqs = []
            for i in range(20):
                dt = 20 if exact else 24
                reqs.append({"key": "p%02d@s1" % i, "id": "p%02d" % i, "category": "c", "kind": "measure", "ok": True,
                             "valid_decode": False, "completion_tokens": 25, "decode_tokens": dt, "decode_s": 1.0,
                             "tok_s": float(dt), "boundary": "exact" if exact else "assumed", "sha256": arm + str(i),
                             "repeat": 0, "seed": 1, "ttft_s": 0.1, "t_wall": 2 * i + (arm == "B")})
            doc = {"schema": "speedbench-result/1", "label": arm, "run_id": arm, "session_id": "s",
                   "prompt_set": {"version": "v", "sha256": "x" * 64},
                   "settings": {"mode": "full", "temperature": 0.7, "seeds": "1", "max_tokens": 256, "effort": "low",
                                "conc": 1, "prime": True, "expected_jobs": 20},
                   "request_template": {}, "summary": {"requests": 20, "failed": 0, "valid_decode": 0}, "requests": reqs}
            paths.append(os.path.join(t, arm + ".json"))
            with open(paths[-1], "w") as f:
                json.dump(doc, f)
        with self.assertRaises(SystemExit) as cm:
            quiet(compare.main, paths)
        self.assertIn("token/time boundary differs", str(cm.exception))

    def test_isolation_is_checked_per_conversation(self):
        rows = [{"kind": "measure", "ok": True, "conv": "c%d" % i, "turn": 1, "repeat": 0, "seed": None,
                 "prompt_tokens": 1000, "cached_tokens": 1000 if i == 0 else 0, "ttft_s": 0.1, "total_s": 1.0}
                for i in range(8)]
        ag = agent.summarize(rows)
        self.assertAlmostEqual(ag["turn1_cached_frac"], 0.125)            # the average looked fine
        self.assertEqual(ag["isolation"], "FAILED in 1 of 8 conversations: c0")
        rows[0]["cached_tokens"] = 16                                     # a template preamble block: still a miss
        self.assertTrue(agent.summarize(rows)["isolation"].startswith("verified"))
        rows[3]["cached_tokens"] = None
        self.assertIn("UNVERIFIED for c3", agent.summarize(rows)["isolation"])


class FinalGate(unittest.TestCase):
    def test_step_rule_needs_complete_coverage(self):
        # 40 prompts x 25 seeds, T>0, step rule pre-registered; one 20-token output per prompt has no engine telemetry
        rng = random.Random(9)
        docs = []
        for arm in ("A", "B"):
            reqs = []
            for i in range(40):
                for s in range(25):
                    gap = s == i % 25
                    st = 10.0 * math.exp(rng.gauss(0, 0.002)) * (0.98 if arm == "B" and not gap else 1.0)
                    r = {"key": "p%02d@s%d" % (i, s), "id": "p%02d" % i, "category": "c%d" % (i % 4),
                         "kind": "measure", "ok": True, "valid_decode": not gap, "decode_tokens": 20 if gap else 200,
                         "tok_s": 3.0 / st * 1000 * math.exp(rng.gauss(0, 0.002)), "sha256": arm + str(i) + str(s),
                         "repeat": 0, "seed": s, "boundary": "exact", "ttft_s": 0.1, "t_wall": 2 * (25 * i + s) + (arm == "B")}
                    if not gap:
                        r.update({"step_ms": st, "step_ms_server": st, "tokens_per_step": 3.0, "draft_per_step": 3.0,
                                  "steps_source": "metrics"})
                    reqs.append(r)
            docs.append({"label": arm, "run_id": arm, "session_id": "s", "prompt_set": {"version": "v", "sha256": "x" * 64},
                         "settings": {"mode": "full", "temperature": 0.7, "seeds": "0-24", "max_tokens": 256,
                                      "effort": "low", "conc": 1, "prime": True, "preregister": "step", "tps_margin": 2.0},
                         "summary": {"requests": 1000, "failed": 0, "valid_decode": 960}, "requests": reqs})
        out = compare.analyse(docs[:1], docs[1:], ARGS)
        self.assertEqual(out["decode"]["pairs_dropped"], 0)               # decode keeps the 20-token outputs
        sr = out["step_rule"]
        self.assertFalse(sr["applies"])
        self.assertTrue(any(nm.startswith("complete coverage") and ok is False for nm, ok, _ in sr["checks"]) or
                        any(nm.startswith("server-side") and ok is False for nm, ok, _ in sr["checks"]))
        _, txt = quiet(compare.report, out, docs[:1], docs[1:], [], [], ARGS)
        self.assertNotIn("DECISION: the step-time verdict", txt)
        # also when the client step time is complete but a decision metric is not
        for d in docs:
            for r in d["requests"]:
                r.setdefault("step_ms", 10.0)
                r.setdefault("step_ms_server", 10.0)
        out = compare.analyse(docs[:1], docs[1:], ARGS)
        checks = {nm: (ok, why) for nm, ok, why in out["step_rule"]["checks"]}
        cov = checks["complete coverage (every pair of the decode analysis)"]
        self.assertFalse(cov[0])
        self.assertIn("tokens_per_step misses 40", cov[1])
        self.assertFalse(out["step_rule"]["applies"])

    def test_failed_first_turns_are_no_isolation_evidence(self):
        rows = []
        for i in range(8):
            for t in (1, 2, 3):
                rows.append({"kind": "measure", "ok": t > 1, "conv": "c%d" % i, "turn": t, "repeat": 0, "seed": None,
                             "prompt_tokens": 1000 * t, "cached_tokens": 900 * (t - 1), "ttft_s": 0.1, "total_s": 1.0})
        iso = agent.summarize(rows)["isolation"]
        self.assertTrue(iso.startswith("UNVERIFIED for all conversations"), iso)
        a, b = agent_docs([-0.05] * 8)
        for d in (a, b):
            d["summary"]["agent"]["isolation"] = iso
        out = compare.analyse([a], [b], ARGS)
        self.assertTrue(out["agent"]["ttft_cached_verdict"].startswith("CACHE ISOLATION UNVERIFIED: B FASTER"),
                        out["agent"]["ttft_cached_verdict"])
