"""Engine counters, decomposition, session handling, the step-time rule, load/longctx modes and the registry."""
import contextlib
import io
import json
import math
import os
import random
import tempfile
import unittest

from speedbench import compare, engines, load, longctx, registry, run, stats as S
from tests.test_e2e_mock import ROOT, Mock


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        res = f(*a, **k)
    return res, out.getvalue()


class Counters(unittest.TestCase):
    TEXT = ('# TYPE vllm:spec_decode_num_drafts_total counter\n'
            'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 10\n'
            'vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 5\n'
            'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 20\n'
            'vllm:iteration_tokens_total_bucket{le="1"} 3\n'
            'vllm:iteration_tokens_total_count{engine="0"} 40\n'
            'vllm:request_decode_time_seconds_sum{engine="0"} 1.5\n'
            'sglang:spec_accept_length{x="y"} 3.2\n')

    def test_parse_and_canonical(self):
        c = engines.canonical(engines.parse_prometheus(self.TEXT))
        self.assertEqual(c["spec_steps"], 15)            # summed over engines
        self.assertEqual(c["accepted"], 20)
        self.assertEqual(c["iterations"], 40)
        self.assertEqual(c["decode_s"], 1.5)
        self.assertNotIn("draft_tokens", c)              # absent stays absent, never 0
        d = engines.delta({"spec_steps": 5, "decode_s": 1.0, "running": 0}, {"spec_steps": 15, "decode_s": 1.5,
                                                                              "running": 1})
        self.assertEqual(d, {"spec_steps": 10, "decode_s": 0.5, "running": 1})

    def test_per_request_fields(self):
        sg = {"sglext": {"spec_tokens_details": {"spec_verify_ct": 21, "spec_num_proposed_drafts": 105,
                                                 "spec_num_correct_drafts": 87}}}
        self.assertEqual(engines.per_request(sg), {"spec_steps": 21, "draft_tokens": 105, "accepted": 87,
                                                   "source": "response"})
        g = {"step_stats": {"steps": 9, "draft_tokens": 27, "accepted_tokens": 14, "decode_s": 0.3}}
        self.assertEqual(engines.per_request(g)["accepted"], 14)
        self.assertIsNone(engines.per_request({}))

    def test_detect(self):
        self.assertEqual(engines.detect({"models": {"data": [{"owned_by": "gr8"}]}}), "gr8")
        self.assertEqual(engines.detect({"models": {"data": [{"owned_by": "vllm"}]}}), "vllm")
        self.assertEqual(engines.detect({}, {"iterations": 3}), "vllm")

    def test_decomposition_identity(self):
        r = {"ok": True, "completion_tokens": 101, "first_chunk_tokens": 1, "last_chunk_tokens": 101,
             "boundary": "exact", "decode_s": 2.0, "n_steps": 50, "n_chunks": 51,
             "spec": {"spec_steps": 40, "draft_tokens": 120, "accepted": 60, "decode_s": 1.9, "finished": 1}}
        run.derive(r)
        self.assertAlmostEqual(r["tokens_per_step"] / r["step_ms"] * 1000, r["tok_s"], places=9)
        self.assertAlmostEqual(r["tokens_per_step"], 2.5)
        self.assertAlmostEqual(r["accept_rate"], 0.5)
        self.assertAlmostEqual(r["step_ms_server"], 1.9 * 1000 / 40)
        self.assertEqual(r["steps_source"], "metrics")


class Sessions(unittest.TestCase):
    def mk(self, n=60, runs=1, shocks=None, luck_sd=0.04, effect=0.0, resid=0.0, seed=1):
        rng = random.Random(seed)
        cats = {"k%02d" % i: "c%d" % (i % 3) for i in range(n)}
        luck = {k: rng.gauss(0, luck_sd) for k in cats}           # fixed per prompt: same text on every boot
        A, B = [], []
        for r in range(runs):
            sh = shocks[r] if shocks else 0.0
            A.append({k: 3.8 + sh + rng.gauss(0, 0.001) for k in cats})
            B.append({k: 3.8 + sh + effect + luck[k] + rng.gauss(0, 0.001) + (resid if r % 2 else -resid)
                      for k in cats})
        return A, B, cats

    def test_boots_are_not_extra_luck_samples(self):
        A1, B1, cats = self.mk(runs=1)
        A2, B2, _ = self.mk(runs=4)
        se1 = S.paired(A1, B1, cats, n_boot=0, shared_boot_sd=0.001)["overall"]["se_prompt"]
        se4 = S.paired(A2, B2, cats, n_boot=0, shared_boot_sd=0.001)["overall"]["se_prompt"]
        self.assertGreater(se4, 0.8 * se1)                       # four boots do not shrink the luck term

    def test_interleaved_sessions_estimate_residual(self):
        A, B, cats = self.mk(runs=4, shocks=[0.0, 0.02, -0.01, 0.015], luck_sd=0.003, resid=0.004)
        r = S.paired(A, B, cats, n_boot=0, shared_boot_sd=0.001)
        b = r["boot"]["A"]
        self.assertEqual(b["source"], "shared sessions: anova+prior")
        self.assertGreater(b["sigma_session_anova"], 0.003)      # +-0.4 % per session -> session SD ~0.46 %
        self.assertGreater(r["overall"]["se"], 0.001)              # prior alone would give 0.05 %


class EndToEndModes(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="speedbench-modes-")

    def p(self, n):
        return os.path.join(self.t, n)

    def test_gr8_counters_at_end_and_step_rule(self):
        a, b = Mock("--flavor", "gr8", "--step-ms", "2"), Mock("--flavor", "gr8", "--step-ms", "1.8", "--traj-sd", "0",
                                                                "--tail-p", "0", "--early-eos", "0")
        try:
            paths, _ = quiet(run.main, ["--arm", "A=" + a.url, "--arm", "B=" + b.url, "--mode", "quick",
                                        "--categories", "prose,code", "--metrics", "--preregister", "step",
                                        "--out", self.p("s-"), "--quiet", "--warmup", "1", "--warmup-prefill", "512"])
            doc = json.load(open(paths[0], encoding="utf-8"))
            r = [x for x in doc["requests"] if x["kind"] == "measure"][0]
            self.assertEqual(r["steps_source"], "metrics")
            self.assertEqual(doc["arm"]["engine"], "gr8")
            self.assertIsNotNone(r.get("step_ms_server"))
            out, txt = quiet(compare.main, paths)
            self.assertTrue(out["step_rule"]["declared"])
            self.assertIn("STEP-TIME RULE", txt)
            self.assertNotIn("CROSS-SESSION", txt)
            self.assertLess(out["step_ms"]["overall"]["ci"][1], 0)
        finally:
            a.stop()
            b.stop()

    def test_cross_session_is_flagged(self):
        m = Mock("--step-ms", "1")
        try:
            for lab in ("x", "y"):
                quiet(run.main, ["--url", m.url, "--label", lab, "--mode", "quick", "--categories", "prose",
                                 "--out", self.p(lab + ".json"), "--quiet", "--warmup", "1", "--warmup-prefill", "0",
                                 "--host-compacted", "yes" if lab == "x" else "unknown"])
            out, txt = quiet(compare.main, [self.p("x.json"), self.p("y.json")])
            self.assertTrue(out["verdict"].startswith("CROSS-SESSION"))
            self.assertIn("CROSS-SESSION COMPARISON", txt)
            self.assertIn("operator metadata host_compacted differs", txt)
        finally:
            m.stop()

    def test_sglang_per_request_stats(self):
        m = Mock("--flavor", "sglang", "--step-ms", "1")
        try:
            path, _ = quiet(run.main, ["--url", m.url, "--mode", "quick", "--categories", "code", "--metrics",
                                       "--out", self.p("sg.json"), "--quiet", "--warmup", "1", "--warmup-prefill", "0"])
            doc = json.load(open(path[0], encoding="utf-8"))
            self.assertEqual(doc["arm"]["stats_field"], "return_spec_tokens_details")
            r = [x for x in doc["requests"] if x["kind"] == "measure"][0]
            self.assertEqual(r["steps_source"], "response")
            self.assertEqual(r["steps_engine"], r["steps_stream"])
        finally:
            m.stop()

    def test_load_and_mix(self):
        m = Mock("--flavor", "gr8", "--proposed", "--step-ms", "2")
        try:
            path, _ = quiet(load.main, ["--url", m.url, "--conc", "1,4", "--duration", "3", "--metrics", "--quiet",
                                        "--mix", "agentic", "--mix-duration", "3", "--long-tokens", "4096", "--short-every", "0.5",
                                        "--short-delay", "0.2", "--warmup", "1", "--out", self.p("load.json")])
            doc = json.load(open(path, encoding="utf-8"))
            l1, l4 = doc["levels"]
            self.assertGreater(l4["aggregate_tok_s"], 1.5 * l1["aggregate_tok_s"])     # batching pays
            self.assertLess(l4["per_stream"]["median_tok_s"], l1["per_stream"]["median_tok_s"])
            self.assertAlmostEqual(l4["engine"]["batch_occupancy"], 4, delta=0.6)
            self.assertGreater(doc["mix"]["short"]["n"], 0)
            self.assertIsNotNone(doc["mix"]["long_by_state"]["alone"]["tok_s"])
            self.assertEqual(doc["meta"]["engine"], "gr8")
        finally:
            m.stop()

    def test_longctx_cold_and_cached(self):
        m = Mock("--step-ms", "1", "--max-model-len", "20000")
        try:
            path, txt = quiet(longctx.main, ["--url", m.url, "--contexts", "4k,8k,32k", "--decode-contexts", "8k",
                                             "--reps", "1", "--decode-tokens", "64", "--quiet",
                                             "--out", self.p("lc.json")])
            doc = json.load(open(path, encoding="utf-8"))
            self.assertEqual(doc["settings"]["skipped_over_limit"], [32768])
            rows = {r["context"]: r for r in doc["table"]}
            self.assertLess(rows[8192]["ttft_cached_s"], rows[8192]["ttft_cold_s"])
            self.assertGreater(rows[8192]["cached_tokens"], 7000)
            self.assertAlmostEqual(rows[8192]["prompt_tokens"] / 8192, 1, delta=0.02)    # sized per document
            self.assertIsNotNone(rows[8192]["decode_tok_s"])
            self.assertIsNone(rows[4096].get("decode_tok_s"))
            cold = [r for r in doc["requests"] if r["kind"] == "cold"]
            self.assertTrue(all((r.get("cached_tokens") or 0) == 0 for r in cold))      # salt defeats the cache
        finally:
            m.stop()


class Registry(unittest.TestCase):
    def test_append_and_readme(self):
        t = tempfile.mkdtemp()
        m = Mock("--step-ms", "1")
        try:
            reg = os.path.join(t, "results", "registry.jsonl")
            paths, _ = quiet(run.main, ["--url", m.url, "--label", "r1", "--mode", "quick", "--categories", "prose",
                                        "--out", os.path.join(t, "r1.json"), "--quiet", "--warmup", "1",
                                        "--warmup-prefill", "0", "--quant", "q4", "--registry", reg])
            quiet(registry.main, ["add", paths[0], "--registry", reg])          # duplicate: ignored
            self.assertEqual(len(registry.load(reg)), 1)
            e = registry.load(reg)[0]
            self.assertEqual(e["quant"], "q4")
            self.assertEqual(e["engine"], "vllm")
            readme = os.path.join(t, "README.md")
            with open(readme, "w") as f:
                f.write("x\n%s\nold\n%s\ny\n" % (registry.BEGIN, registry.END))
            quiet(registry.main, ["table", "--registry", reg, "--readme", readme])
            body = open(readme).read()
            self.assertIn("| r1 |", body)
            self.assertNotIn("old", body)
            self.assertTrue(body.startswith("x\n") and body.endswith("y\n"))
        finally:
            m.stop()


class StepRule(unittest.TestCase):
    ARGS = dict(boot_prior=1.0, shared_prior=0.1, alt_prior=2.0, alpha=0.05, decide_on="tok", tps_margin=2.0)

    def docs(self, step_b=0.99, tps_b=1.0, luck=0.004, interleave=True, fail=0, seed=3):
        import argparse
        rng = random.Random(seed)
        out = []
        for arm in ("A", "B"):
            reqs = []
            for i in range(30):
                tps = 3.0 * (tps_b if arm == "B" else 1.0) * math.exp(rng.gauss(0, luck))
                st = 10.0 * (step_b if arm == "B" else 1.0) * math.exp(rng.gauss(0, 0.002))
                reqs.append({"key": "p%02d" % i, "id": "p%02d" % i, "category": "c%d" % (i % 3), "kind": "measure",
                             "ok": i >= fail, "valid_decode": True, "tok_s": tps / st * 1000, "step_ms": st,
                             "step_ms_server": st, "tokens_per_step": tps, "draft_per_step": 3.0,
                             "steps_source": "metrics", "sha256": "%s%d" % (arm, i), "repeat": 0, "boundary": "exact",
                             "ttft_s": 0.1, "t_wall": (2 * i + (arm == "B")) if interleave else (i + (40 if arm == "B" else 0))})
            out.append({"label": arm, "run_id": arm, "session_id": "s1", "settings": {"mode": "full", "preregister": "step",
                        "tps_margin": 2.0}, "summary": {}, "requests": reqs})
        return out, argparse.Namespace(**self.ARGS)

    def rule(self, **kw):
        (a, b), args = self.docs(**kw)
        out = compare.analyse([a], [b], args)
        return out, out["step_rule"]

    def test_pure_step_lever_decides(self):
        out, sr = self.rule()
        self.assertTrue(sr["applies"], sr["checks"])
        self.assertTrue(sr["verdict"].startswith("B FASTER"))
        self.assertEqual(sr["field"], "step_ms_server")
        self.assertEqual(sr["margin_source"], "pre-registered")

    def test_acceptance_loss_cannot_flip_a_slower_tok_s(self):
        out, sr = self.rule(tps_b=0.985, luck=0.0005)
        self.assertEqual(out["verdict"], "B SLOWER")
        self.assertFalse(sr["applies"])

    def test_acceptance_loss_outweighing_step_gain_does_not_decide(self):
        out, sr = self.rule(tps_b=0.985, luck=0.03)
        self.assertNotIn("B SLOWER", out["verdict"])                 # tok/s inconclusive
        self.assertFalse(sr["applies"], sr["checks"])

    def test_margin_must_be_recorded(self):
        (a, b), args = self.docs()
        del b["settings"]["tps_margin"]
        self.assertFalse(compare.analyse([a], [b], args)["step_rule"]["applies"])

    def test_not_interleaved_and_withheld_gate(self):
        out, sr = self.rule(interleave=False)
        self.assertFalse(sr["applies"])
        self.assertTrue(out["verdict"].startswith("NOT INTERLEAVED"))
        self.assertFalse(out["session_cancels"])
        out, sr = self.rule(fail=3)
        self.assertTrue(out["verdict"].startswith("WITHHELD"))
        self.assertFalse(sr["applies"])


class LoadDetails(unittest.TestCase):
    def test_sizing_counts_the_returned_text(self):
        from unittest.mock import patch
        from speedbench import longdoc

        def fake_count(url, model, text, headers=None):          # density changes along the text
            return int(len(text[:20000]) / 3.0 + len(text[20000:]) / 4.6)
        with patch.object(longdoc, "count_tokens", fake_count):
            doc, n, conv = longdoc.sized("u", "m", 32768, 3.6, seed=1, salt="x")
        self.assertTrue(conv)
        self.assertEqual(n, fake_count(None, None, longdoc.messages(doc, longdoc.QUESTIONS[0])[0]["content"]))
        self.assertAlmostEqual(n / 32768, 1, delta=0.01)

    def test_mix_split_at_state_changes(self):
        by = load.split_by_state([[(0.0, 0), (1.0, 10)]], [(0.1, 0.4, 0.6)])
        self.assertAlmostEqual(by["short_prefilling"][1], 0.3)
        self.assertAlmostEqual(by["short_decoding"][1], 0.2)
        self.assertAlmostEqual(by["alone"][1], 0.5)
        self.assertAlmostEqual(by["alone"][0], 5.0)

    def test_no_per_chunk_usage_no_window_throughput(self):
        m = Mock("--reject-continuous", "--step-ms", "1")
        try:
            t = tempfile.mkdtemp()
            path, _ = quiet(load.main, ["--url", m.url, "--conc", "2", "--duration", "2", "--quiet", "--warmup", "1",
                                        "--out", os.path.join(t, "l.json")])
            lv = json.load(open(path, encoding="utf-8"))["levels"][0]
            self.assertIsNone(lv["aggregate_tok_s"])
            self.assertGreater(lv["block_tok_s"], 0)
        finally:
            m.stop()

    def test_report_ci_is_over_prompts(self):
        from speedbench import report
        rng = random.Random(5)
        reqs = [{"id": "p%d" % i, "category": "c", "kind": "measure", "ok": True, "valid_decode": True,
                 "tok_s": 40 * math.exp(rng.gauss(0, 0.05))} for i in range(20)]
        doc = {"label": "x", "arm": {}, "settings": {"mode": "full"}, "requests": reqs}
        one = io.StringIO()
        report.print_decode(doc, one.write)
        doc["requests"] = reqs * 4                                  # the same trajectories four times
        four = io.StringIO()
        report.print_decode(doc, four.write)
        self.assertEqual(one.getvalue().splitlines()[-4], four.getvalue().splitlines()[-4])


if __name__ == "__main__":
    unittest.main()
