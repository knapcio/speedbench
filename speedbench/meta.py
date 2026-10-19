"""Run metadata shared by every mode: what was measured, on what, in which host state (operator-provided fields
are recorded as given; `unknown` when not given)."""
import datetime
import platform
import socket

from . import __version__, engines

TRISTATE = ("yes", "no", "unknown")


def add_args(ap):
    g = ap.add_argument_group("metadata (recorded in the result and the registry; operator-provided)")
    g.add_argument("--quant", default="unknown", help="quantization, e.g. 'gr8 EXLR8 mix', 'NVFP4', 'FP8'")
    g.add_argument("--engine", default=None, help="engine name (default: detected from /v1/models and /metrics)")
    g.add_argument("--engine-version", default=None, help="engine version (default: from /version when exposed)")
    g.add_argument("--engine-commit", default="unknown", help="engine commit or image tag")
    g.add_argument("--config", dest="config_summary", default="", help="one-line configuration summary (flags, TP, ...)")
    g.add_argument("--hardware", default="unknown", help="e.g. '4x DGX Spark, TP4, CX-7 200G'")
    g.add_argument("--thinking", default="unknown",
                   help="thinking mode as served, e.g. 'on, effort low' (the request's effort is recorded separately)")
    g.add_argument("--host-compacted", choices=TRISTATE, default="unknown",
                   help="memory compacted before the server started (sync; echo 1 > /proc/sys/vm/compact_memory)")
    g.add_argument("--fresh-boot", choices=TRISTATE, default="unknown", help="server started fresh for this session")
    g.add_argument("--clock-policy", default="unknown", help="GPU clock policy, e.g. 'cap 2200 MHz (nvidia-smi -lgc 0,2200)'")
    g.add_argument("--notes", default="", help="free text")
    g.add_argument("--registry", default=None,
                   help="append a summary of this run to this JSONL registry (e.g. results/registry.jsonl)")
    return ap


def collect(a, server_meta, counters=None, mode=None):
    eng = a.engine or engines.detect(server_meta or {}, counters)
    return {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "model": (server_meta or {}).get("model"),
        "quant": a.quant, "engine": eng,
        "engine_version": a.engine_version or engines.engine_version(server_meta or {}) or "unknown",
        "engine_commit": a.engine_commit, "config": a.config_summary, "hardware": a.hardware,
        "thinking": a.thinking, "effort": getattr(a, "effort", None),
        "host_compacted": a.host_compacted, "fresh_boot": a.fresh_boot, "clock_policy": a.clock_policy,
        "notes": a.notes, "tool_version": __version__,
        "client": {"host": socket.gethostname(), "python": platform.python_version(), "platform": platform.platform()},
    }


HOST_FIELDS = ("host_compacted", "fresh_boot", "clock_policy", "engine_commit", "quant", "hardware")
