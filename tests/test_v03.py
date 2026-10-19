"""v0.3 features: multi-turn agent mode, energy per token, quality guard and divergence, T>0 with seeds."""
import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import time
import unittest

from speedbench import agent, compare, guard, power, registry, run, selftest, stats as S
from tests.test_e2e_mock import Mock

FAKE_POWER = "%s -u -c \"import time\nwhile True:\n    print(%%s, flush=True)\n    time.sleep(0.05)\"" % sys.executable


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
        res = f(*a, **k)
    return res, out.getvalue()


class AgentSet(unittest.TestCase):
    def setUp(self):
        self.body, _ = agent.load()

    def test_shape_and_sizes(self):
        convs = self.body["conversations"]
        self.assertEqual(len(convs), 8)
        full = agent.conversations(self.body, "agent")
        quick = agent.conversations(self.body, "agent-quick")
        self.assertEqual(sum(n for _, n in full), 48)
        self.assertEqual(sum(n for _, n in quick), 16)
        self.assertTrue(all(4 <= n <= 8 for _, n in full))
        self.assertTrue(any(c["tools"] for c in convs) and any(not c["tools"] for c in convs))
        self.assertTrue(all(not c["tools"] for c, _ in agent.conversations(self.body, "agent", skip_tools=True)))

    def test_every_request_extends_the_previous(self):
        for conv, n in agent.conversations(self.body, "agent"):
            prev = None
            for t in range(1, n + 1):
                msgs = agent.request_messages(self.body, conv, t, "Session: x")
                self.assertIn(msgs[-1]["role"], ("user", "tool"))
                if prev is not None:
                    self.assertEqual(msgs[:len(prev)], prev)          # prefix property: the cache can serve it
                    self.assertEqual(msgs[len(prev)]["role"], "assistant")
                prev = msgs

    def test_isolation(self):
        a = argparse.Namespace(agent_nonce="n1", isolate="text", temperature=0.0, seeds="1", repeats=1,
                               mode="agent-quick", skip_tools=False)
        job = agent.plan(self.body, a)[1]
        pa = agent.materialize(self.body, job, run.Arm("A", "http://x"), a)
        pb = agent.materialize(self.body, job, run.Arm("B", "http://x"), a)
        self.assertNotEqual(pa["messages"][0], pb["messages"][0])        # salt line per arm
        self.assertEqual(pa["messages"][1:], pb["messages"][1:])
        a.isolate = "field"
        pa = agent.materialize(self.body, job, run.Arm("A", "http://x"), a)
        pb = agent.materialize(self.body, job, run.Arm("B", "http://x"), a)
        self.assertEqual(pa["messages"], pb["messages"])
        self.assertNotEqual(pa["cache_salt"], pb["cache_salt"])
        with self.assertRaises(SystemExit):                              # protected request field
            run.check_extra({"cache_salt": "x"}, "--extra-body")


class AgentEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = tempfile.mkdtemp(prefix="speedbench-agent-")
        cls.m = Mock("--step-ms", "1")

    @classmethod
    def tearDownClass(cls):
        cls.m.stop()

    def ab(self, prefix, *extra):
        # both arms on ONE server: the isolation must keep B from reusing A's cache
        paths, _ = quiet(run.main, ["--arm", "A=" + self.m.url, "--arm", "B=" + self.m.url, "--mode", "agent-quick",
                                    "--out", os.path.join(self.t, prefix), "--quiet", "--warmup", "1",
                                    "--warmup-prefill", "0", "--metrics"] + list(extra))
        return paths, [jload(p) for p in paths]

    def test_text_isolation_cache_and_compare(self):
        paths, docs = self.ab("txt-")
        for d in docs:
            ag = d["summary"]["agent"]
            self.assertEqual(ag["turn1_cached_frac"], 0.0)                 # turn 1 cold in both arms
            self.assertGreater(ag["cached"]["cache_hit_vs_ideal"], 0.9)    # later turns served from the cache
            self.assertLess(ag["cached"]["median_ttft_ms"], ag["cold"]["median_ttft_ms"])
            self.assertEqual(d["settings"]["expected_jobs"], 16)
            self.assertFalse(d["settings"]["prime"])
            r = [x for x in d["requests"] if x["turn"] == 2][0]
            self.assertEqual(r["key"], r["conv"] + "@t2")
        out, txt = quiet(compare.main, paths)
        self.assertIn("AGENT (multi-turn", txt)
        self.assertIn("SCREEN ONLY (agent-quick mode)", out["verdict"])
        # CIs are clustered by conversation: 4 conversations (agent-quick), pairs are turns
        self.assertEqual(out["agent"]["ttft_cached"]["n_prompts"], 4)
        self.assertEqual(out["agent"]["ttft_cached"]["pairs_planned"], 12)
        self.assertEqual(out["agent"]["ttft_cold"]["n_prompts"], 4)
        self.assertTrue(out["agent"]["ttft_cached_verdict"].startswith("SCREEN ONLY"))
        self.assertIsNone(out.get("divergence"))                          # outputs differ by design (salt)

    def test_field_isolation_keeps_text(self):
        paths, docs = self.ab("fld-", "--isolate", "field")
        for d in docs:
            self.assertEqual(d["summary"]["agent"]["turn1_cached_frac"], 0.0)
        out, _ = quiet(compare.main, paths)
        self.assertEqual(out["identical_between_arms"][0], out["identical_between_arms"][1])   # same config, same text

    def test_ignored_cache_salt_is_detected(self):
        m = Mock("--step-ms", "1", "--ignore-cache-salt")
        try:
            paths, _ = quiet(run.main, ["--arm", "A=" + m.url, "--arm", "B=" + m.url, "--mode", "agent-quick",
                                        "--isolate", "field", "--out", os.path.join(self.t, "ign-"), "--quiet",
                                        "--warmup", "1", "--warmup-prefill", "0"])
            b = jload(paths[1])
            self.assertIn("warning", b["summary"]["agent"])                # B found A's turn 1 in the cache
        finally:
            m.stop()


class Power(unittest.TestCase):
    def test_parse_and_integrate(self):
        self.assertEqual(power.parse_line("45.20\n"), 45.2)
        self.assertEqual(power.parse_line("[N/A]"), None)
        self.assertEqual(power.parse_line(""), None)
        self.assertEqual(power.parse_line("10, 20.5"), 30.5)
        t = [i * 0.1 for i in range(41)]
        self.assertAlmostEqual(power.integrate(t, [100.0] * 41, 0.55, 2.55, 1.0), 200.0)
        ramp = [10 * x for x in t]                                        # P = 10 t  ->  integral 5 (t1^2 - t0^2)
        self.assertAlmostEqual(power.integrate(t, ramp, 1.0, 3.0, 1.0), 40.0)
        self.assertIsNone(power.integrate(t, ramp, 3.0, 9.0, 1.0))        # not covered by the trace
        self.assertIsNone(power.integrate([0, 0.1, 5.0], [1, 1, 1], 0.0, 5.0, 1.0))   # a gap inside the window

    def test_sources_are_summed(self):
        s = power.Sampler([FAKE_POWER % 100, FAKE_POWER % 50])
        s.start()
        try:
            t0 = s.clock()
            time.sleep(1.0)
            t1 = s.clock()
            time.sleep(0.3)
            self.assertAlmostEqual(s.mean_power(t0, t1), 150.0, delta=0.5)
        finally:
            s.stop()
        d = s.describe()
        self.assertEqual(len(d["sources"]), 2)
        self.assertGreater(d["sources"][0]["n"], 10)

    def test_run_energy_per_token(self):
        m = Mock("--step-ms", "2")
        try:
            t = tempfile.mkdtemp()
            paths, _ = quiet(run.main, ["--url", m.url, "--mode", "quick", "--categories", "prose", "--quiet",
                                        "--warmup", "1", "--warmup-prefill", "0", "--out", os.path.join(t, "e.json"),
                                        "--power-cmd", FAKE_POWER % 100, "--power-cmd", FAKE_POWER % 50,
                                        "--power-idle", "0.5"])
            doc = jload(paths[0])
            self.assertEqual(doc["settings"]["power"]["kind"], "gpu")
            self.assertIn("excludes CPU", doc["energy"]["note"])
            self.assertAlmostEqual(doc["energy"]["idle_w"], 150, delta=1)
            for r in doc["requests"]:
                if r.get("valid_decode") and r["kind"] == "measure":
                    self.assertAlmostEqual(r["j_per_tok"] * r["tok_s"], 150, delta=8)   # J/token = W / (tok/s)
            e = doc["summary"]["ALL"]["energy"]
            self.assertAlmostEqual(e["decode_mean_w"], 150, delta=2)
            # off by default: no energy fields at all
            paths, _ = quiet(run.main, ["--url", m.url, "--mode", "quick", "--categories", "prose", "--quiet",
                                        "--warmup", "1", "--warmup-prefill", "0", "--out", os.path.join(t, "n.json")])
            doc = jload(paths[0])
            self.assertNotIn("energy", doc)
            self.assertNotIn("power", doc["settings"])
            self.assertFalse(any("t_send_rel" in r or "j_per_tok" in r for r in doc["requests"]))
        finally:
            m.stop()


class Guard(unittest.TestCase):
    def test_graders(self):
        num = {"type": "number", "value": 6912}
        self.assertTrue(guard.grade(num, "1234 + 5678 = **6,912**")[0])
        self.assertFalse(guard.grade(num, "6911")[0])
        self.assertFalse(guard.grade(num, "")[0])
        txt = {"type": "text", "accept": ["jane austen", "austen"]}
        self.assertTrue(guard.grade(txt, "Jane Austen.")[0])
        self.assertTrue(guard.grade(txt, "The author is\n**Austen**")[0])
        self.assertFalse(guard.grade(txt, "Charlotte Bronte")[0])
        case = {"type": "text", "accept": ["HELLO WORLD"], "case": True}
        self.assertTrue(guard.grade(case, "HELLO WORLD")[0])
        self.assertFalse(guard.grade(case, "hello world")[0])
        js = {"type": "json", "value": [1, 3, 5]}
        self.assertTrue(guard.grade(js, "```json\n[1, 3, 5]\n```")[0])
        self.assertTrue(guard.grade(js, "Sorted: [1,3,5]")[0])
        self.assertFalse(guard.grade(js, "[1, 5, 3]")[0])
        self.assertTrue(guard.grade({"type": "text", "accept": ["2024-02-29"]}, "2024-02-29")[0])

    def test_shipped_items_grade_their_own_answers(self):
        body = guard.load()
        self.assertEqual(len(body["items"]), 30)
        for it in body["items"]:
            g = it["grader"]
            ans = {"number": str(g.get("value")), "text": (g.get("accept") or [""])[0],
                   "json": json.dumps(g.get("value"))}[g["type"]]
            self.assertTrue(guard.grade(g, ans)[0], it["id"])

    def test_compare_flags_a_drop(self):
        ga = {"items": [{"id": "i%d" % k, "pass": True, "sha256": "a%d" % k} for k in range(30)], "truncated": 0}
        gb = {"items": [{"id": "i%d" % k, "pass": k >= 4, "sha256": "b%d" % k} for k in range(30)], "truncated": 0}
        r = guard.compare_guard(ga, gb)
        self.assertTrue(r["drop"])
        self.assertEqual(r["passed_B"], 26)
        self.assertLess(r["p_one_sided"], 0.1)
        r = guard.compare_guard(ga, ga)
        self.assertFalse(r["drop"])
        self.assertEqual(r["identical_outputs"], 30)
        self.assertAlmostEqual(guard.mcnemar_one_sided(1, 1), 0.75)

    def test_first_divergence_position(self):
        a = {"text": {"reasoning": "aaaa bbbb ", "content": "cccc dddd"}, "chunks": [
            [1, 5, 1, "r"], [2, 5, 2, "r"], [3, 5, 3, "c"], [4, 4, 4, "c"]], "completion_tokens": 4}
        b = {"text": {"reasoning": "aaaa bbbb ", "content": "cccc XXXX"}}
        self.assertEqual(guard.first_divergence(a, b), 3)                 # differs in the 4th token
        self.assertIsNone(guard.first_divergence(a, a))

    def test_end_to_end_guard_and_divergence(self):
        a, b = Mock("--step-ms", "1", "--diverge", "0.5"), Mock("--step-ms", "1", "--numerics", "2", "--diverge", "0.5",
                                                                  "--break-answers", "0.3")
        try:
            t = tempfile.mkdtemp()
            paths, _ = quiet(run.main, ["--arm", "A=" + a.url, "--arm", "B=" + b.url, "--mode", "quick", "--quiet",
                                        "--categories", "prose,code", "--warmup", "1", "--warmup-prefill", "0",
                                        "--quality-guard", "--out", os.path.join(t, "g-")])
            da = jload(paths[0])
            self.assertEqual(da["guard"]["passed"], 30)
            self.assertIn("text", [r for r in da["requests"] if r["kind"] == "measure"][0])   # implied --keep-text
            out, txt = quiet(compare.main, paths)
            self.assertTrue(out["guard"][0]["drop"])
            self.assertIn("FLAG: B LOST ITEMS", txt)
            dv = out["divergence"]
            self.assertEqual(dv["differing"], dv["n"])
            self.assertGreater(dv["median"], 16)                          # numerics-like: late divergence
            self.assertIn("late divergence", txt)
        finally:
            a.stop()
            b.stop()


class Sampling(unittest.TestCase):
    """T>0: pairs per (prompt, seed), CI over prompts including the sampling variance"""

    def docs(self, n=40, seeds=(1, 2, 3), luck=0.06, between=0.01, effect=0.0, short=(), rng_seed=4):
        rng = random.Random(rng_seed)
        out = []
        pe = {i: rng.gauss(0, between) for i in range(n)}
        for arm in ("A", "B"):
            reqs = []
            for i in range(n):
                for s in seeds:
                    v = 40 * math.exp(rng.gauss(0, luck) + (effect + pe[i] if arm == "B" else 0))
                    reqs.append({"key": "p%02d@s%d" % (i, s), "id": "p%02d" % i, "category": "c%d" % (i % 4),
                                 "kind": "measure", "ok": True, "valid_decode": (arm, i, s) not in short,
                                 "decode_tokens": 5 if (arm, i, s) in short else 200,
                                 "tok_s": v, "sha256": "%s%d%d" % (arm, i, s), "repeat": 0, "seed": s,
                                 "boundary": "exact", "ttft_s": 0.1, "t_wall": 2 * i + (arm == "B")})
            out.append({"label": arm, "run_id": arm, "session_id": "s", "prompt_set": {"version": "v", "sha256": "x" * 64},
                        "settings": {"mode": "full", "temperature": 0.7, "seeds": ",".join(map(str, seeds)),
                                     "max_tokens": 256, "effort": "low", "conc": 1, "prime": True},
                        "summary": {"requests": len(reqs), "failed": 0, "valid_decode": len(reqs)},
                        "requests": reqs})
        args = argparse.Namespace(boot_prior=1.0, shared_prior=0.1, alt_prior=2.0, alpha=0.05, decide_on="tok",
                                  tps_margin=2.0)
        return out, args

    def test_pairs_per_prompt_seed(self):
        (a, b), args = self.docs(short={("B", 3, 2), ("A", 7, 1)})
        out = compare.analyse([a], [b], args)
        d = out["decode"]
        self.assertEqual(d["n_prompts"], 40)                  # a short seed drops the pair, not the prompt
        self.assertEqual(d["pairs_dropped"], 2)
        self.assertEqual(d["pairs_planned"], 120)
        self.assertFalse(out["verdict"].startswith("WITHHELD"))
        sm = out["sampling"]
        self.assertEqual(sm["seeds"], 3)
        self.assertGreater(sm["sd_pair"], 0.05)                # 6 % per trajectory and arm -> ~8.5 % per pair
        self.assertLess(sm["mde_by_seeds"][12], sm["mde_by_seeds"][3])
        self.assertAlmostEqual(sm["mde_by_seeds"][3], d["mde80"], places=9)   # consistent with the verdict's MDE
        _, txt = quiet(compare.report, out, [a], [b], [], [], args)
        self.assertIn("T>0 (T=0.7", txt)
        self.assertIn("(prompt, seed) pairs dropped", txt)

    def test_many_dropped_pairs_withhold(self):
        (a, b), args = self.docs(short={("B", i, 1) for i in range(20)})
        out = compare.analyse([a], [b], args)
        self.assertTrue(out["verdict"].startswith("WITHHELD"))

    def test_t0_keeps_whole_prompt_rule(self):
        cats = {"p%d@v%d" % (i, v): "all" for i in range(6) for v in range(2)}
        cl = {k: k.split("@")[0] for k in cats}
        A = [{k: 3.8 for k in cats}]
        B = [{k: 3.8 for k in cats}]
        del B[0]["p0@v1"]
        self.assertEqual(S.paired(A, B, cats, cluster=cl, n_boot=0)["excluded"], ["p0"])
        r = S.paired(A, B, cats, cluster=cl, n_boot=0, partial=True)
        self.assertEqual((r["excluded"], r["pairs_dropped"]), ([], 1))

    def test_selftest_coverage_with_seeds(self):
        sc, _ = selftest.scenarios()
        kw = dict(sc)["T>0, full, 3 seeds per prompt, interleaved"]
        r = selftest.scenario("t", 150, 7, **kw)
        self.assertGreater(r["coverage0"], 0.9)
        self.assertLess(r["fp"], 0.1)


class RegistryAgent(unittest.TestCase):
    def test_agent_rows(self):
        doc = {"schema": "speedbench-result/1", "label": "ag", "settings": {"mode": "agent", "max_tokens": 160},
               "meta": {"mode": "agent"}, "summary": {"ALL": {"geomean_tok_s": 40.0, "energy": {"decode_tok_per_j": 0.2}},
                                                      "agent": {"cold": {"median_ttft_ms": 900}, "cached": {
                                                          "median_ttft_ms": 120, "cache_hit_vs_ideal": 0.98}}},
               "energy": {"kind": "gpu"}}
        t = tempfile.mkdtemp()
        p = os.path.join(t, "r.json")
        with open(p, "w") as f:
            json.dump(doc, f)
        text = registry.table([registry.entry(doc, p)])
        self.assertIn("Multi-turn agent", text)
        self.assertIn("0.20 (gpu)", text)
        self.assertNotIn("#### Decode", text)


if __name__ == "__main__":
    unittest.main()
