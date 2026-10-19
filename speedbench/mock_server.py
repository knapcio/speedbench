"""Mock OpenAI-compatible server that simulates speculative decoding with trajectory luck.

Each request's "trajectory" is a deterministic function of (prompt, numerics id): the engine's numerics decide
the T=0 text, and the text decides how predictable it is for the drafter. A change of `--numerics` therefore
puts every prompt on a new trajectory with a new acceptance rate (the luck the benchmark must average out),
while `--step-ms` changes the cost of an engine step (the real speed of a lever).

One engine thread runs the steps for all requests, like a real continuous-batching server: a decode step serves
every decoding request at once (its cost grows with the batch, `--batch-cost`, and with the context,
`--ctx-cost`), and prompts are prefilled in chunks (`--prefill-chunk`) between decode steps, so a long prefill
slows the streams that are decoding. A token-block prefix cache makes repeated prefixes cheap. The stream
follows vLLM: reasoning first (`reasoning_content`), then `content`; one SSE chunk per engine step; usage on the
last chunk and, if asked, cumulative usage on every chunk. `/metrics` imitates one engine (`--flavor`):

  vllm    vllm:* counters updated every step, iteration histogram, request decode-time histogram
  gr8     vllm:* names, but spec counters and request timing added when a request ENDS (as gr8's S2 shim does);
          no /version; `--proposed` adds the counters of docs/gr8-step-counters-proposal.md
  sglang  sglang:* counters; per-request `sglext.spec_tokens_details` when the request asks for it

The defaults give a per-prompt luck of ~10 % at 256 tokens, about twice what we measured on a real engine
(4-5 %), so the mock exaggerates the problem the benchmark solves.

  python3 -m speedbench mock --port 8399 --step-ms 20 --numerics 1 [--flavor gr8]
"""
import argparse
import collections
import hashlib
import json
import math
import queue
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = ("the of and to in is that for it as with on was by be this are from at or an which but have not "
         "one all their can more will about would there what when out so up into time some could them "
         "these other than then now only its also two first way even new because any where most").split()
BLOCK_CHARS = 256          # prefix-cache block: 256 characters of the serialised prompt (~64 tokens)


def _u(*parts):
    """deterministic uniform(0,1) from the parts"""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return (int.from_bytes(h[:8], "big") + 0.5) / 2.0 ** 64


def _normal(*parts):
    u1, u2 = _u(*parts, "a"), _u(*parts, "b")
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


class Req:
    def __init__(self, key, traj, p_acc, eos_at, n_reason, prompt_tokens, cached, rng, text_rngs=None, answer=None):
        self.key, self.traj, self.p_acc, self.eos_at, self.n_reason = key, traj, p_acc, eos_at, n_reason
        self.prompt_tokens, self.cached, self.rng = prompt_tokens, cached, rng
        self.text_rngs = text_rngs        # (shared rng, own rng, token position where the own text starts)
        self.answer = answer              # guard item: the content to send after the reasoning
        self.todo = prompt_tokens - cached           # prompt tokens still to prefill
        self.made = 0
        self.out = queue.Queue()
        self.cancelled = False
        self.t_arrive = time.monotonic()
        self.t_first = None
        self.st = {"rounds": 0, "drafted": 0, "accepted": 0}


class Sim:
    def __init__(self, a):
        self.a = a
        self.cv = threading.Condition()
        self.pending = collections.deque()           # waiting for (more) prefill
        self.active = []                             # decoding
        self.c = collections.Counter()               # metric counters
        self.cache = set()
        self.boot = math.exp(a.boot_sd * _normal("boot", a.boot_seed)) if a.boot_sd else 1.0
        self.t_start = time.monotonic()
        self.turn = 0
        threading.Thread(target=self.loop, daemon=True).start()

    def accept_rate(self, prompt_key, traj):
        """per-trajectory acceptance probability per draft position: prompt base x lognormal luck (+ tail)"""
        a = self.a
        base = 0.55 + 0.3 * _u("base", prompt_key)                     # prompt content: 0.55..0.85
        luck = a.traj_sd * _normal("luck", prompt_key, traj)
        if _u("tail", prompt_key, traj) < a.tail_p:                      # rare mode switch (e.g. list/loop)
            luck += a.tail_size * (1 if _u("tailsign", prompt_key, traj) < 0.5 else -1)
        p = 1 - (1 - base) * math.exp(-luck * 4)
        return min(0.97, max(0.05, p))

    def speed(self):
        return self.boot * (1 + self.a.drift_per_min * (time.monotonic() - self.t_start) / 60)

    # ------------------------------------------------------------------ prefix cache (token blocks)
    def blocks(self, text):
        h = hashlib.sha256()
        out = []
        for i in range(0, len(text) - BLOCK_CHARS + 1, BLOCK_CHARS):
            h.update(text[i:i + BLOCK_CHARS].encode())
            out.append(h.copy().hexdigest()[:20])
        return out

    def lookup(self, text, prompt_tokens):
        if self.a.no_prefix_cache:
            return 0, []
        bl = self.blocks(text)
        n = 0
        with self.cv:
            for b in bl:
                if b not in self.cache:
                    break
                n += 1
        cached = min(prompt_tokens - 1, n * BLOCK_CHARS // 4)
        return max(0, cached), bl

    def remember(self, bl):
        with self.cv:
            if len(self.cache) > 400000:
                self.cache.clear()
            self.cache.update(bl)

    # ------------------------------------------------------------------ engine loop
    def submit(self, r):
        with self.cv:
            self.pending.append(r)
            self.c["prompt_tokens"] += r.prompt_tokens
            self.c["cached_tokens"] += r.cached
            self.c["prefix_queries"] += r.prompt_tokens
            self.c["prefix_hits"] += r.cached
            self.cv.notify()

    def loop(self):
        while True:
            with self.cv:
                while not self.pending and not self.active:
                    self.cv.wait()
                for r in [x for x in self.active if x.cancelled]:
                    self.active.remove(r)
                while self.pending and self.pending[0].cancelled:
                    self.pending.popleft()
                # alternate prefill chunks and decode steps while both are waiting
                do_prefill = bool(self.pending) and (not self.active or self.turn % 2 == 0)
                self.turn += 1
                job = self.pending[0] if do_prefill else None
                batch = list(self.active) if not do_prefill else None
            if job is not None:
                self.prefill_chunk(job)
            elif batch:
                self.decode_step(batch)

    def prefill_chunk(self, r):
        a = self.a
        n = min(r.todo, a.prefill_chunk) if r.todo > 0 else 0
        pos = r.prompt_tokens - r.todo                       # tokens before this chunk
        # linear cost + attention over the context so far (quadratic over the whole prompt)
        cost_ms = a.prefill_ms_per_1k * n / 1000 + a.prefill_quad * n * (pos + n / 2) / 1e6
        time.sleep(max(cost_ms, 0.2) / 1000 / self.speed())
        r.todo -= n
        with self.cv:
            self.c["iterations"] += 1
            self.c["iteration_tokens"] += n
        if r.todo > 0:
            return
        with self.cv:
            self.pending.popleft()
            self.c["iteration_tokens"] += 1
        self.emit(r, 1, first=True)                           # the prefill step yields one token
        if r.made < r.eos_at and not r.cancelled:
            with self.cv:
                self.active.append(r)

    def decode_step(self, batch):
        a = self.a
        ctx = max(r.prompt_tokens + r.made for r in batch)
        jit = 1 + a.jitter * (2 * random.random() - 1)
        width = 1 + a.width_cost * a.k
        cost = a.step_ms * width * (1 + a.batch_cost * (len(batch) - 1)) * (1 + a.ctx_cost * ctx / 1000) * jit
        t0 = time.monotonic()
        time.sleep(cost / 1000 / self.speed())
        made = acc_sum = 0
        for r in batch:
            acc = 0
            while acc < a.k and r.rng.random() < r.p_acc:
                acc += 1
            k = min(1 + acc, r.eos_at - r.made)
            if r.answer is not None and r.made < r.n_reason:
                k = min(k, r.n_reason - r.made)          # the answer goes out in its own content chunk
            r.st["rounds"] += 1
            r.st["drafted"] += a.k
            r.st["accepted"] += acc
            acc_sum += acc
            made += k
            self.emit(r, k)
        with self.cv:
            self.c["iterations"] += 1
            self.c["iteration_tokens"] += made
            self.c["engine_steps"] += 1
            self.c["engine_step_s"] += time.monotonic() - t0
            self.c["engine_step_sessions"] += len(batch)
            if a.flavor != "gr8":                              # vLLM: spec counters every step
                self.c["drafts"] += len(batch)
                self.c["draft_tokens"] += a.k * len(batch)
                self.c["accepted"] += acc_sum
            for r in batch:
                if r.made >= r.eos_at and r in self.active:
                    self.active.remove(r)

    def emit(self, r, k, first=False):
        now = time.monotonic()
        chan = "reasoning_content" if r.made < r.n_reason else "content"
        if r.text_rngs:              # --diverge: the text follows a shared trajectory up to a per-prompt position
            shared, own, at = r.text_rngs
            text = " ".join(WORDS[(shared if r.made + j < at else own).randrange(len(WORDS))] for j in range(k)) + " "
        else:
            text = " ".join(WORDS[r.rng.randrange(len(WORDS))] for _ in range(k)) + " "
        if r.answer is not None and chan == "content":
            text = r.answer
        r.made += k
        with self.cv:
            self.c["gen_tokens"] += k
            if first:
                r.t_first = now
                self.c["ttft_s"] += now - r.t_arrive
                self.c["ttft_n"] += 1
            done = r.made >= r.eos_at
            if done:
                self.finish(r, now)
                if r in self.active:
                    self.active.remove(r)
        r.out.put(("chunk", text, chan, r.made))
        if done:
            r.out.put(("done",))

    def finish(self, r, now):
        """request-end accounting (cv held)"""
        c = self.c
        c["decode_s"] += now - r.t_first
        c["prefill_s"] += r.t_first - r.t_arrive
        c["finished"] += 1
        c["req_gen_tokens"] += r.made
        if self.a.flavor == "gr8":                              # gr8: the engine's "m" line, added at the end
            c["drafts"] += r.st["rounds"]
            c["draft_tokens"] += r.st["drafted"]
            c["accepted"] += r.st["accepted"]

    # ------------------------------------------------------------------ /metrics
    def metrics_text(self):
        with self.cv:
            c = dict(self.c)
            running = len(self.active)
        f = self.a.flavor
        lines = []

        def m(name, v, kind="counter"):
            lines.append("# TYPE %s %s" % (name, kind))
            lines.append('%s{engine="0",model_name="%s"} %s' % (name, self.a.model, repr(float(v))))
        if f == "sglang":
            m("sglang:prompt_tokens_total", c.get("prompt_tokens", 0))
            m("sglang:generation_tokens_total", c.get("gen_tokens", 0))
            m("sglang:cached_tokens_total", c.get("cached_tokens", 0))
            m("sglang:num_requests_total", c.get("finished", 0))
            m("sglang:time_to_first_token_seconds_sum", c.get("ttft_s", 0), "histogram")
            m("sglang:time_to_first_token_seconds_count", c.get("ttft_n", 0), "histogram")
            m("sglang:num_running_reqs", running, "gauge")
            m("sglang:spec_accept_length", 1 + self.a.k * 0.6, "gauge")
            return "\n".join(lines) + "\n"
        m("vllm:num_requests_running", running, "gauge")
        m("vllm:spec_decode_num_drafts_total", c.get("drafts", 0))
        m("vllm:spec_decode_num_draft_tokens_total", c.get("draft_tokens", 0))
        m("vllm:spec_decode_num_accepted_tokens_total", c.get("accepted", 0))
        m("vllm:generation_tokens_total", c.get("gen_tokens", 0))
        m("vllm:prompt_tokens_total", c.get("prompt_tokens", 0))
        m("vllm:prompt_tokens_cached_total", c.get("cached_tokens", 0))
        m("vllm:prefix_cache_queries_total", c.get("prefix_queries", 0))
        m("vllm:prefix_cache_hits_total", c.get("prefix_hits", 0))
        m("vllm:time_to_first_token_seconds_sum", c.get("ttft_s", 0))
        m("vllm:time_to_first_token_seconds_count", c.get("ttft_n", 0))
        m("vllm:request_decode_time_seconds_sum", c.get("decode_s", 0))
        m("vllm:request_prefill_time_seconds_sum", c.get("prefill_s", 0))
        m("vllm:request_generation_tokens_sum", c.get("req_gen_tokens", 0))
        m("vllm:request_generation_tokens_count", c.get("finished", 0))
        if f == "vllm":
            m("vllm:request_decode_time_seconds_count", c.get("finished", 0))
            m("vllm:iteration_tokens_total_sum", c.get("iteration_tokens", 0))
            m("vllm:iteration_tokens_total_count", c.get("iterations", 0))
        if f == "gr8" and self.a.proposed:
            m("g6:s2_rounds_total", c.get("engine_steps", 0))
            m("g6:s2_round_seconds_total", c.get("engine_step_s", 0))
            m("g6:s2_round_sessions_total", c.get("engine_step_sessions", 0))
        return "\n".join(lines) + "\n"


SIM = None
GUARD = {}          # guard item text -> correct answer (the mock answers guard items like a working model)


def _load_guard():
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "speedbench-guard-v1.json")
    try:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)["items"]
    except (OSError, ValueError, KeyError):
        return {}
    out = {}
    for it in items:
        g = it["grader"]
        ans = {"number": lambda: str(g["value"]), "text": lambda: g["accept"][0],
               "json": lambda: json.dumps(g["value"])}[g["type"]]()
        out[it["messages"][-1]["content"]] = ans
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, obj, ctype="application/json"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Server", "speedbench-mock")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        a = SIM.a
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": a.model, "object": "model",
                                                                 "owned_by": a.flavor, "max_model_len": a.max_model_len}]})
        if self.path.startswith("/version") and a.flavor == "vllm":
            return self._send(200, {"version": "mock-1", "numerics": a.numerics, "step_ms": a.step_ms})
        if self.path.startswith("/server_info") and a.flavor == "sglang":
            return self._send(200, {"version": "mock-sglang-1"})
        if self.path.startswith("/metrics"):
            return self._send(200, SIM.metrics_text().encode(), "text/plain")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path.startswith("/tokenize"):
            text = body.get("prompt") or json.dumps(body.get("messages") or [], sort_keys=True)
            return self._send(200, {"count": max(1, len(text) // 4), "max_model_len": SIM.a.max_model_len})
        if not self.path.startswith("/v1/chat/completions"):
            return self._send(404, {"error": "not found"})
        if SIM.a.reject_continuous and (body.get("stream_options") or {}).get("continuous_usage_stats"):
            return self._send(400, {"error": "unknown field continuous_usage_stats"})
        if not body.get("stream"):
            return self._send(400, {"error": "mock supports stream=true only"})
        self.stream(body)

    def stream(self, body):
        a = SIM.a
        msgs = body.get("messages") or []
        text = json.dumps(msgs, sort_keys=True)
        key = hashlib.sha256(text.encode()).hexdigest()[:16]
        salt = body.get("cache_salt")
        cache_text = text if not salt or a.ignore_cache_salt else "salt:%s|%s" % (salt, text)
        prompt_tokens = max(1, len(text) // 4)
        if prompt_tokens > a.max_model_len:
            return self._send(400, {"error": {"message": "prompt of %d tokens exceeds max_model_len %d" % (
                prompt_tokens, a.max_model_len)}})
        temp = float(body.get("temperature") or 0)
        seed = body.get("seed")
        if temp == 0:
            traj = "n%s" % a.numerics
            if a.nondeterministic and _u("nd", key, time.monotonic_ns()) < a.nondeterministic:
                traj += "-%d" % time.monotonic_ns()
        else:      # sampled: the seed and the numerics both matter (different numerics desynchronise draws)
            traj = "n%s-s%s" % (a.numerics, seed if seed is not None else time.monotonic_ns())
        max_tokens = int(body.get("max_tokens") or 256)
        eos_at = max_tokens
        if _u("eos", key, traj) < a.early_eos:
            eos_at = max(2, int(max_tokens * (0.05 + 0.5 * _u("eosat", key, traj))))
        if max_tokens == 1:
            eos_at = 1
        n_reason = int(eos_at * (0.4 + 0.4 * _u("rfrac", key)))
        answer = None
        last = (msgs[-1].get("content") or "") if msgs else ""
        if last in GUARD and max_tokens > 1:
            # a guard item: reason a little, then answer (wrong for a fraction --break-answers of the items)
            n_reason = min(n_reason, 24)
            eos_at = min(max_tokens, n_reason + 1)
            ok = _u("break", key, a.numerics) >= a.break_answers
            answer = GUARD[last] if ok else "I am not sure."
        text_rngs = None
        if a.diverge is not None and temp == 0:
            at = int(eos_at * a.diverge * (0.3 + 1.4 * _u("divpos", key)))
            text_rngs = (random.Random("%s|shared" % key), random.Random("%s|%s|own" % (key, traj)), at)
        so = body.get("stream_options") or {}
        cont, include_usage = so.get("continuous_usage_stats"), so.get("include_usage")
        cached, bl = SIM.lookup(cache_text, prompt_tokens)
        r = Req(key, traj, SIM.accept_rate(key, traj), eos_at, n_reason, prompt_tokens, cached,
                random.Random("%s|%s" % (key, traj)), text_rngs, answer)
        SIM.submit(r)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        rid = "chatcmpl-mock"
        while True:
            item = r.out.get()
            if item[0] == "done":
                break
            _, txt, chan, made = item
            ch = {"id": rid, "object": "chat.completion.chunk", "model": a.model,
                  "choices": [{"index": 0, "delta": {chan: txt}, "finish_reason": None}]}
            if cont:
                ch["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": made,
                               "total_tokens": prompt_tokens + made}
            if not self._emit(ch):
                r.cancelled = True
                return
        SIM.remember(bl)
        made = r.made
        fin = "length" if eos_at >= max_tokens else "stop"
        self._emit({"id": rid, "object": "chat.completion.chunk", "model": a.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": fin}]})
        if include_usage:
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": made, "total_tokens": prompt_tokens + made,
                     "prompt_tokens_details": None if a.no_cache_report else {"cached_tokens": cached},
                     "completion_tokens_details": {"reasoning_tokens": min(made, n_reason)}
                     if a.report_reasoning else None}
            last = {"id": rid, "object": "chat.completion.chunk", "model": a.model, "choices": [], "usage": usage}
            if a.flavor == "gr8" and a.proposed and body.get("return_step_stats"):
                usage["step_stats"] = {"steps": r.st["rounds"], "draft_tokens": r.st["drafted"],
                                       "accepted_tokens": r.st["accepted"],
                                       "decode_s": round(time.monotonic() - r.t_first, 6)}
            if a.flavor == "sglang" and body.get("return_spec_tokens_details") and r.st["rounds"]:
                last["sglext"] = {"spec_tokens_details": {
                    "spec_verify_ct": r.st["rounds"], "spec_num_proposed_drafts": r.st["drafted"],
                    "spec_num_correct_drafts": r.st["accepted"],
                    "spec_accept_length": (made - 1) / r.st["rounds"]}}
            self._emit(last)
        self._raw(b"data: [DONE]\n\n")

    def _emit(self, obj):
        return self._raw(("data: " + json.dumps(obj) + "\n\n").encode())

    def _raw(self, b):
        try:
            self.wfile.write(b)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False


def parser():
    ap = argparse.ArgumentParser(prog="speedbench mock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--flavor", choices=["vllm", "gr8", "sglang"], default="vllm",
                    help="which engine's /metrics and per-request fields to imitate")
    ap.add_argument("--proposed", action="store_true",
                    help="gr8 flavour: also export the proposed step counters and usage.step_stats")
    ap.add_argument("--step-ms", type=float, default=20.0, help="engine step time (the lever)")
    ap.add_argument("--numerics", default="1", help="numerics id: changing it re-draws every trajectory")
    ap.add_argument("--k", type=int, default=3, help="draft length")
    ap.add_argument("--width-cost", type=float, default=0.05, help="relative step cost per draft position")
    ap.add_argument("--batch-cost", type=float, default=0.08, help="relative step cost per extra stream in a step")
    ap.add_argument("--ctx-cost", type=float, default=0.003, help="relative step cost per 1K tokens of context")
    ap.add_argument("--traj-sd", type=float, default=0.02, help="per-trajectory acceptance luck (log-odds-ish)")
    ap.add_argument("--tail-p", type=float, default=0.04, help="probability of a heavy-tail trajectory")
    ap.add_argument("--tail-size", type=float, default=0.06)
    ap.add_argument("--jitter", type=float, default=0.02, help="uniform +- step-time jitter")
    ap.add_argument("--early-eos", type=float, default=0.05, help="fraction of trajectories that stop early")
    ap.add_argument("--prefill-ms-per-1k", type=float, default=50.0, help="linear prefill cost")
    ap.add_argument("--prefill-quad", type=float, default=0.2, help="attention prefill cost, ms per (1K tokens)^2")
    ap.add_argument("--prefill-chunk", type=int, default=2048, help="prefill tokens per engine step")
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--boot-sd", type=float, default=0.0, help="per-boot speed shock SD (log)")
    ap.add_argument("--boot-seed", default="0")
    ap.add_argument("--drift-per-min", type=float, default=0.0)
    ap.add_argument("--nondeterministic", type=float, default=0.0, help="P(T=0 request lands on a new trajectory)")
    ap.add_argument("--reject-continuous", action="store_true", help="reject continuous_usage_stats with 400")
    ap.add_argument("--report-reasoning", action="store_true", help="report usage reasoning_tokens")
    ap.add_argument("--ignore-cache-salt", action="store_true", help="ignore the cache_salt request field")
    ap.add_argument("--no-cache-report", action="store_true",
                    help="omit usage.prompt_tokens_details (like vLLM without --enable-prompt-tokens-details)")
    ap.add_argument("--break-answers", type=float, default=0.0,
                    help="fraction of quality-guard items answered wrongly (per prompt and numerics id)")
    ap.add_argument("--diverge", type=float, default=None,
                    help="T=0 text shared by all numerics ids up to a per-prompt position (mean fraction of the "
                         "output, e.g. 0.5), then trajectory-specific (exercises the divergence summary)")
    return ap


def serve(argv=None, ready=None):
    global SIM, GUARD
    a = parser().parse_args(argv)
    SIM = Sim(a)
    GUARD = _load_guard()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.daemon_threads = True
    if ready is not None:
        ready.append(srv)
    srv.serve_forever()


def main(argv=None):
    serve(argv)


if __name__ == "__main__":
    main()
