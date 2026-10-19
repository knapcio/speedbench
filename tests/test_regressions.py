"""Regression tests for the round-2 review findings (measurement contract, duplicates, exclusions, overrides)."""
import copy
import io
import itertools
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from speedbench import client, compare, run, stats as S


def stream(events, done=True):
    payload = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + ("data: [DONE]\n\n" if done else "")
    ticks = itertools.count()
    with patch("urllib.request.urlopen", return_value=io.BytesIO(payload.encode())):
        r = client.stream_chat("http://fixture", {}, clock=lambda: next(ticks))
    r["ok"] = True
    return run.derive(r)


def text(cum=None):
    e = {"choices": [{"delta": {"content": "x"}}]}
    if cum is not None:
        e["usage"] = {"completion_tokens": cum}
    return e


def finish(total, reason="stop"):
    return {"choices": [{"delta": {}, "finish_reason": reason}], "usage": {"completion_tokens": total}}


class Contract(unittest.TestCase):
    def test_truncated_stream_fails(self):
        with self.assertRaises(client.RequestFailed):
            stream([text(1), text(65)], done=False)

    def test_exact_boundaries_ignore_trailing_tokens(self):
        r = stream([text(1), text(100), finish(108)])      # 8 tokens (e.g. EOS, hidden) after the last text event
        self.assertEqual(r["boundary"], "exact")
        self.assertEqual(r["decode_tokens"], 99)
        self.assertAlmostEqual(r["tok_s"], 99.0)

    def test_first_burst_counted(self):
        r = stream([text(8), text(16), text(256), finish(256, "length")])
        self.assertEqual(r["decode_tokens"], 248)

    def test_assumed_boundary_is_fallback_formula(self):
        r = stream([text(), text(), finish(50)])
        self.assertEqual(r["boundary"], "assumed")
        self.assertEqual(r["decode_tokens"], 49)

    def test_midstream_exception_is_a_failed_request(self):
        class Boom(io.BytesIO):
            def __iter__(self):
                yield b'data: {"choices": [{"delta": {"content": "x"}}]}\n'
                yield b"\n"
                raise TimeoutError("read timed out")
        a = SimpleNamespace(temperature=0.0, top_p=None, effort="low", effort_field="template", timeout=5)
        arm = run.Arm("A", "http://fixture", "m")
        with patch("urllib.request.urlopen", return_value=Boom()):
            r = run.one(arm, {"messages": [{"role": "user", "content": "hi"}]}, a, 16)
        self.assertFalse(r["ok"])
        self.assertIn("interrupted", r["error"])


class Overrides(unittest.TestCase):
    def test_protected_fields(self):
        for extra in ({"temperature": 1}, {"max_tokens": 2048}, {"chat_template_kwargs": {"reasoning_effort": "max"}}):
            with self.assertRaises(SystemExit):
                run.check_extra(extra, "--extra-body")
        run.check_extra({"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}, "--extra-body")

    def test_prime_stays_one_token(self):
        a = SimpleNamespace(temperature=0.0, top_p=None, effort="low", effort_field="template")
        arm = run.Arm("A", "http://x", "m", extra={"top_k": 5})
        body = run.build_body(arm, {"messages": [{"role": "user", "content": "q"}]}, a, 1)
        self.assertEqual(body["max_tokens"], 1)
        self.assertEqual(body["chat_template_kwargs"], {"reasoning_effort": "low"})


class Statistics(unittest.TestCase):
    def test_failed_variant_drops_whole_prompt(self):
        cats = {"p%d@v%d" % (i, v): "all" for i in range(6) for v in range(2)}
        cl = {k: k.split("@")[0] for k in cats}
        A = [{k: 3.8 for k in cats}]
        B = [{k: 3.8 + (0.1 if k.endswith("v0") else -0.1) for k in cats}]
        for i in range(3):
            del B[0]["p%d@v1" % i]                       # the negative variants of three prompts failed
        r = S.paired(A, B, cats, cluster=cl, n_boot=0)
        self.assertEqual(r["excluded"], ["p0", "p1", "p2"])
        self.assertAlmostEqual(r["overall"]["est"], 0.0, places=9)

    def test_frozen_category_weights(self):
        cats = {"a%d" % i: "a" for i in range(10)}
        cats.update({"b%d" % i: "b" for i in range(10)})
        A = [{k: 3.8 for k in cats}]
        B = [{k: 3.8 + (0.02 if k.startswith("a") else 0.0) for k in cats}]
        for i in range(5):
            del B[0]["b%d" % i]
        r = S.paired(A, B, cats, n_boot=0)
        self.assertAlmostEqual(r["overall"]["est"], 0.01, places=9)   # 50/50 planned weights, not 10/5


class CompareInputs(unittest.TestCase):
    def doc(self, label, path, cont=True, run_id=None, extra=None, val=100.0):
        reqs = []
        for i in range(10):
            reqs.append({"key": "p%d" % i, "id": "p%d" % i, "category": "c", "kind": "measure", "ok": True,
                         "valid_decode": True, "tok_s": val + i, "sha256": "h%d" % i, "repeat": 0,
                         "boundary": "exact" if cont else "assumed", "ttft_s": 0.1})
        d = {"schema": "speedbench-result/1", "label": label, "run_id": run_id or label, "block": None,
             "prompt_set": {"sha256": "x", "version": "v"},
             "settings": {"mode": "full", "max_tokens": 256, "temperature": 0, "prime": True, "conc": 1,
                          "expected_jobs": 10, "effort": "low"},
             "request_template": {"temperature": 0, "max_tokens": 256, "extra": 1, **({"top_k": extra} if extra else {})},
             "summary": {"requests": 10, "failed": 0, "valid_decode": 10}, "requests": reqs}
        with open(path, "w") as f:
            json.dump(d, f)
        return path

    def setUp(self):
        self.t = tempfile.mkdtemp()

    def p(self, n):
        return os.path.join(self.t, n)

    def test_duplicate_run_rejected(self):
        a = self.doc("A", self.p("a.json"))
        b = self.doc("B", self.p("b.json"), val=101)
        with self.assertRaises(SystemExit):
            compare.main(["--a", a, a, "--b", b, self.doc("B2", self.p("b2.json"), val=101)])

    def test_boundary_mismatch_rejected(self):
        a = self.doc("A", self.p("a.json"))
        b = self.doc("B", self.p("b.json"), cont=False, val=101)
        with self.assertRaises(SystemExit):
            compare.main([a, b])

    def test_effective_request_mismatch_rejected(self):
        a = self.doc("A", self.p("a.json"), extra=1)
        b = self.doc("B", self.p("b.json"), extra=2, val=101)
        with self.assertRaises(SystemExit):
            compare.main([a, b])


if __name__ == "__main__":
    unittest.main()
