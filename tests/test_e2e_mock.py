"""End-to-end: the real client against mock OpenAI-compatible servers that simulate speculative decoding."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from speedbench import client, compare, run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Mock:
    def __init__(self, *args):
        self.port = free_port()
        self.p = subprocess.Popen([sys.executable, "-m", "speedbench", "mock", "--port", str(self.port)] + list(args),
                                  cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.url = "http://127.0.0.1:%d" % self.port
        for _ in range(100):
            try:
                urllib.request.urlopen(self.url + "/v1/models", timeout=1).read()
                return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("mock did not start")

    def stop(self):
        self.p.terminate()
        self.p.wait()


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="speedbench-test-")
        cls.A = Mock("--step-ms", "2.0", "--numerics", "1")
        cls.B = Mock("--step-ms", "1.7", "--numerics", "2")      # faster steps, different trajectories
        # every comparison is an interleaved session (both arms alternate request by request): a CPU stall on a
        # loaded machine then hits both arms alike instead of shifting one whole run (the mock's steps are 2 ms)
        cls.files = {}
        for pre, other, url in (("same-", "A2", cls.A.url), ("ab-", "B", cls.B.url)):
            with open(os.devnull, "w") as dn:
                old, sys.stdout = sys.stdout, dn
                try:
                    pa, pb = run.main(["--arm", "A=" + cls.A.url, "--arm", "%s=%s" % (other, url), "--mode", "quick",
                                       "--out", os.path.join(cls.tmp, pre), "--quiet", "--metrics"])
                finally:
                    sys.stdout = old
            cls.files[pre + "A"], cls.files[other] = pa, pb
        cls.files["A"] = cls.files["ab-A"]
        cls.files["A1"] = cls.files["same-A"]

    @classmethod
    def tearDownClass(cls):
        cls.A.stop()
        cls.B.stop()

    def analyse(self, a, b):
        out_json = os.path.join(self.tmp, "cmp.json")
        with open(os.devnull, "w") as dn:
            old = sys.stdout
            sys.stdout = dn
            try:
                return compare.main([self.files[a], self.files[b], "--json", out_json])
            finally:
                sys.stdout = old

    def test_result_file(self):
        with open(self.files["A"], encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["schema"], "speedbench-result/1")
        self.assertEqual(doc["summary"]["failed"], 0)
        r = [x for x in doc["requests"] if x["kind"] == "measure"][0]
        for k in ("tok_s", "ttft_s", "completion_tokens", "reasoning_tokens", "first_chunk_tokens", "step_ms",
                  "sha256", "prime", "spec"):
            self.assertIn(k, r)
        self.assertEqual(r["first_chunk_tokens"], 1)
        self.assertEqual(doc["summary"]["sentinel"]["identical_text"], 2)
        # chunk count matches the engine's step counter (one SSE chunk per step)
        self.assertEqual(r["steps_stream"], r["spec"]["spec_steps"])

    def test_detects_faster_engine(self):
        out = self.analyse("A", "B")
        self.assertTrue(out["session_cancels"])
        self.assertTrue(out["verdict"].endswith("B FASTER"))
        self.assertEqual(out["identical_between_arms"][0], 0)
        st = out["step_ms"]["overall"]
        self.assertLess(st["ci"][1], 0)            # step time clearly lower

    def test_same_config_covers_zero(self):
        out = self.analyse("A1", "A2")
        self.assertTrue(out["session_cancels"])
        lo, hi = out["decode"]["overall"]["ci"]
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 0)
        self.assertEqual(out["identical_between_arms"][0], out["identical_between_arms"][1])


class Fallbacks(unittest.TestCase):
    def test_rejected_continuous_usage(self):
        m = Mock("--reject-continuous", "--step-ms", "1")
        try:
            path = os.path.join(tempfile.mkdtemp(), "r.json")
            run.main(["--url", m.url, "--mode", "quick", "--categories", "prose", "--out", path, "--quiet",
                      "--warmup", "1", "--sentinels", "0"])
            doc = json.load(open(path))
            self.assertFalse(doc["arm"]["continuous_usage"])
            self.assertEqual(doc["summary"]["failed"], 0)
            r = doc["requests"][0]
            self.assertIsNone(r["first_chunk_tokens"])
            self.assertGreater(r["tok_s"], 0)
        finally:
            m.stop()

    def test_two_arm_interleaved(self):
        a, b = Mock("--step-ms", "1"), Mock("--step-ms", "1", "--numerics", "9")
        try:
            pre = os.path.join(tempfile.mkdtemp(), "ab-")
            paths = run.main(["--arm", "X=" + a.url, "--arm", "Y=%s,model=mock-model,extra={\"top_k\": 1}" % b.url,
                              "--mode", "quick", "--categories", "code", "--out", pre, "--quiet", "--warmup", "1"])
            self.assertEqual(len(paths), 2)
            x, y = (json.load(open(p, encoding="utf-8")) for p in paths)
            self.assertEqual(y["arm"]["extra_body"], {"top_k": 1})
            # ABBA: the first job runs X then Y, the second Y then X
            self.assertEqual(len(x["requests"]), len(y["requests"]))
            self.assertEqual(x["session_id"], y["session_id"])
            with open(os.devnull, "w") as dn:
                old, sys.stdout = sys.stdout, dn
                try:
                    out = compare.main(paths)
                finally:
                    sys.stdout = old
            self.assertTrue(out["shared_boots"])
            self.assertIn("treatment difference extra_body", " ".join(compare.check_inputs(
                [compare.load(paths[0])], [compare.load(paths[1])], False)[1]))
        finally:
            a.stop()
            b.stop()


class Canned(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    BODY = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(Canned.BODY)


class Parsing(unittest.TestCase):
    def serve(self, body):
        Canned.BODY = body
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Canned)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, "http://127.0.0.1:%d" % srv.server_address[1]

    def test_reasoning_key_tool_calls_and_crlf(self):
        ev = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"reasoning": "think "}}]},
            {"choices": [{"index": 0, "delta": {"reasoning": "more"}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "x1", "function": {"name": "read_file", "arguments": "{\"pa"}}]}}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "th\": \"a\"}"}}]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 9,
                                      "completion_tokens_details": {"reasoning_tokens": 4}}},
        ]
        body = b"".join(b"data: " + json.dumps(e).encode() + b"\r\n\r\n" for e in ev) + b"data: [DONE]\r\n\r\n"
        srv, url = self.serve(body)
        try:
            r = client.stream_chat(url, {"model": "m", "messages": []})
        finally:
            srv.shutdown()
        self.assertEqual(r["completion_tokens"], 9)
        self.assertEqual(r["reasoning_tokens"], 4)
        self.assertEqual(r["finish_reason"], "tool_calls")
        self.assertEqual(r["text"]["tool_calls"], [{"name": "read_file", "arguments": "{\"path\": \"a\"}"}])
        self.assertEqual(r["text"]["reasoning"], "think more")
        self.assertEqual(r["n_steps"], 3)

    def test_missing_usage_is_an_error(self):
        body = b'data: {"choices": [{"index": 0, "delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n'
        srv, url = self.serve(body)
        try:
            with self.assertRaises(client.RequestFailed):
                client.stream_chat(url, {"model": "m", "messages": []})
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
