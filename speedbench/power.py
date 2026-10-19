"""Optional power sampling (`--power-cmd`): energy per output token and tokens per joule.

Each source is a shell command that keeps running and prints one power reading in watts per line, e.g.

  nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits -lms 200      (local GPU 0)
  ssh node2 nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits -lms 200
  my-wall-meter --watts --interval 0.2                                                 (any watts-per-line tool)

Several sources are summed (one per GPU or node of a multi-node server). A line's reading is the sum of the numbers
on it ("N/A" and other non-numeric lines are skipped and counted), so a line must contain watt values only. Lines
are timestamped on arrival with the client's clock (the same clock as the request timings); readings are
interpolated linearly between samples and integrated over each request's time windows. Alignment is therefore
good to about one sampling interval plus the transport delay (ssh), and a single short request's energy is noisy;
aggregates over many requests are not. Local commands run under a pseudo-terminal so that line-buffered output
arrives promptly; a source whose lines arrive in bursts is flagged (`buffered_frac`).

What the number means depends on the source: GPU-reported power (nvidia-smi power.draw) covers the GPU board
only (on unified-memory systems such as GB10, what the driver attributes to the GPU), not the CPU, memory, network,
fans or power-supply losses of the rest of the system. Only a wall-power meter measures the whole system; declare
it with `--power-kind wall`. Polling the GPU driver can itself cost a little speed: compare speed only between
runs that sampled power the same way.
"""
import bisect
import os
import re
import shlex
import signal
import statistics
import subprocess
import threading
import time

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_line(line):
    """watts on one line (sum of its numbers), or None"""
    s = line.strip()
    if not s or "N/A" in s.upper():
        return None
    nums = _NUM.findall(s)
    if not nums:
        return None
    return sum(float(x) for x in nums)


class Source:
    def __init__(self, cmd, clock):
        self.cmd, self.clock = cmd, clock
        self.t, self.w = [], []
        self.skipped = 0
        self.error = None
        self.proc = None
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        master = None
        try:
            import pty
            master, slave = pty.openpty()
            out = slave
        except (ImportError, OSError):
            out = subprocess.PIPE
        try:
            self.proc = subprocess.Popen(self.cmd, shell=True, stdout=out, stderr=subprocess.DEVNULL,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            self.error = "could not start: %r" % (e,)
            return
        if master is not None:
            os.close(slave)
            self.thread = threading.Thread(target=self._read_fd, args=(master,), daemon=True)
        else:
            self.thread = threading.Thread(target=self._read_pipe, daemon=True)
        self.thread.start()

    def _add(self, line):
        v = parse_line(line)
        now = self.clock()
        with self.lock:
            if v is None:
                self.skipped += 1
            else:
                self.t.append(now)
                self.w.append(v)

    def _read_fd(self, fd):
        buf = b""
        try:
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:          # EIO when the child closes the terminal
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self._add(line.decode(errors="replace"))
        finally:
            os.close(fd)

    def _read_pipe(self):
        for line in self.proc.stdout:
            self._add(line.decode(errors="replace"))

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is not None and not self.t:
            self.error = "exited with code %s before printing a reading" % self.proc.returncode
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
        if self.thread:
            self.thread.join(timeout=2)

    def snapshot(self):
        with self.lock:
            return list(self.t), list(self.w)


def integrate(t, w, t0, t1, max_gap):
    """joules of a piecewise-linear power trace over [t0, t1]; None if the trace does not cover the window"""
    if not t or t1 <= t0:
        return None
    if t[0] > t0 + max_gap or t[-1] < t1 - max_gap:
        return None
    # every gap between consecutive readings that touches the window (including the readings just before t0 and
    # just after t1) must be short: a source that stalled and resumed leaves no evidence inside the gap
    i0 = max(0, bisect.bisect_right(t, t0) - 1)
    i1 = min(len(t) - 1, bisect.bisect_left(t, t1))
    if any(t[k + 1] - t[k] > max_gap for k in range(i0, i1)):
        return None

    def at(x):
        i = bisect.bisect_left(t, x)
        if i <= 0:
            return w[0]
        if i >= len(t):
            return w[-1]
        ta, tb = t[i - 1], t[i]
        return w[i - 1] + (w[i] - w[i - 1]) * (x - ta) / (tb - ta) if tb > ta else w[i]
    lo, hi = bisect.bisect_right(t, t0), bisect.bisect_left(t, t1)
    xs = [t0] + t[lo:hi] + [t1]
    vs = [at(t0)] + w[lo:hi] + [at(t1)]
    return sum((xs[k + 1] - xs[k]) * (vs[k] + vs[k + 1]) / 2 for k in range(len(xs) - 1))


class Sampler:
    """all sources of one run; times are seconds of `clock` (time.perf_counter, like the request timings)"""

    def __init__(self, cmds, kind="gpu", clock=time.perf_counter):
        self.kind = kind
        self.clock = clock
        self.sources = [Source(c, clock) for c in cmds]
        self.origin = clock()

    def start(self, wait_s=10.0):
        for s in self.sources:
            s.start()
        deadline = self.clock() + wait_s          # wait for the first reading of every source (ssh may be slow)
        while self.clock() < deadline:
            if all(s.snapshot()[0] or s.error or (s.proc and s.proc.poll() is not None) for s in self.sources):
                break
            time.sleep(0.05)
        missing = [s.cmd for s in self.sources if not s.snapshot()[0]]
        if missing:
            self.stop()
            raise SystemExit("power source(s) printed no reading within %.0f s: %s" % (wait_s, missing))

    def stop(self):
        for s in self.sources:
            s.stop()

    def max_gap(self, t):
        if len(t) < 3:
            return 2.0
        return max(1.0, 5 * statistics.median(b - a for a, b in zip(t, t[1:])))

    def energy(self, t0, t1):
        """joules summed over the sources, or None if any source does not cover [t0, t1]"""
        tot = 0.0
        for s in self.sources:
            t, w = s.snapshot()
            e = integrate(t, w, t0, t1, self.max_gap(t))
            if e is None:
                return None
            tot += e
        return tot

    def mean_power(self, t0, t1):
        e = self.energy(t0, t1)
        return None if e is None or t1 <= t0 else e / (t1 - t0)

    def describe(self, keep_samples=True):
        out = {"kind": self.kind, "sources": []}
        for s in self.sources:
            t, w = s.snapshot()
            gaps = [b - a for a, b in zip(t, t[1:])]
            d = {"cmd": s.cmd, "n": len(t), "skipped_lines": s.skipped, "error": s.error,
                 "mean_w": round(statistics.fmean(w), 2) if w else None,
                 "median_interval_ms": round(statistics.median(gaps) * 1000, 1) if gaps else None,
                 # lines that arrive almost together were buffered by the command: their timestamps are wrong
                 "buffered_frac": round(sum(g < 0.005 for g in gaps) / len(gaps), 3) if gaps else None}
            if keep_samples:
                d["samples"] = [[round(x - self.origin, 3), round(y, 2)] for x, y in zip(t, w)]
            out["sources"].append(d)
        warn = [d["cmd"] for d in out["sources"] if (d["buffered_frac"] or 0) > 0.2]
        if warn:
            out["warning"] = ("readings arrive in bursts (output buffered by the command): timestamps are unreliable "
                              "for %s; make the command line-buffered (e.g. `stdbuf -oL`, or `ssh -tt`)" % warn)
        return out


def add_args(ap):
    g = ap.add_argument_group("energy (optional; off unless --power-cmd is given)")
    g.add_argument("--power-cmd", action="append", default=[],
                   help="shell command printing watts per line, repeat for several GPUs/nodes (summed), e.g. "
                        + shlex.quote("nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits -lms 200"))
    g.add_argument("--power-kind", choices=["gpu", "wall", "other"], default="gpu",
                   help="what the sources measure: gpu (board power only, the default) or wall (whole system)")
    g.add_argument("--power-idle", type=float, default=5.0,
                   help="seconds of idle power measured after the warm-up (0 = none)")
    g.add_argument("--power-no-samples", action="store_true", help="do not store the raw samples in the result")
    return g


def start(a):
    """a started Sampler, or None when power sampling is off"""
    if not getattr(a, "power_cmd", None):
        return None
    s = Sampler(a.power_cmd, a.power_kind)
    s.start()
    return s


def idle(sampler, seconds):
    if not sampler or not seconds:
        return None
    t0 = sampler.clock()
    time.sleep(seconds)
    w = sampler.mean_power(t0, sampler.clock())
    return None if w is None else round(w, 2)


NOTE = {"gpu": "GPU-reported power: excludes CPU, memory, network, fans and power-supply losses of the rest of the "
               "system (not wall power)",
        "wall": "wall power of the measured system (as declared by the operator)",
        "other": "power as measured by the operator's sources (see the commands)"}


def annotate(recs, sampler, origin=None):
    """add per-request energy (c1 runs: requests do not overlap) from t_send_rel/ttft/decode windows"""
    origin = sampler.origin if origin is None else origin
    for r in recs:
        ts = r.get("t_send_rel")
        if ts is None or not r.get("ok"):
            continue
        t0 = origin + ts
        e = sampler.energy(t0, t0 + r["total_s"])
        if e is not None:
            r["energy_j"] = round(e, 3)
            if r.get("completion_tokens"):
                r["j_per_tok_request"] = round(e / r["completion_tokens"], 5)
        if r.get("ttft_s") is not None and r.get("decode_s"):
            ed = sampler.energy(t0 + r["ttft_s"], t0 + r["ttft_s"] + r["decode_s"])
            if ed is not None and r.get("decode_tokens"):
                r["decode_energy_j"] = round(ed, 3)
                r["j_per_tok"] = ed / r["decode_tokens"]


def summarize(recs, idle_w=None):
    """aggregate J/token and tok/J over requests (sum of energy / sum of tokens)"""
    rs = [r for r in recs if r.get("decode_energy_j") is not None and r.get("valid_decode")]
    if not rs:
        return None
    ej = sum(r["decode_energy_j"] for r in rs)
    tok = sum(r["decode_tokens"] for r in rs)
    dt = sum(r["decode_s"] for r in rs)
    out = {"n": len(rs), "decode_j_per_tok": round(ej / tok, 5), "decode_tok_per_j": round(tok / ej, 4),
           "decode_mean_w": round(ej / dt, 2) if dt else None}
    rq = [r for r in rs if r.get("energy_j") is not None]
    if rq:
        out["request_j_per_tok"] = round(sum(r["energy_j"] for r in rq) / sum(r["completion_tokens"] for r in rq), 5)
    if idle_w is not None and dt:
        dyn = ej - idle_w * dt
        out["decode_j_per_tok_above_idle"] = round(dyn / tok, 5)
    return out
