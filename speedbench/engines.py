"""Engine counters for the speed decomposition: tok/s = tokens per step x steps per second.

Three sources, best first:
1. per-request fields in the response (exact per request, valid at any concurrency):
   SGLang `sglext.spec_tokens_details` (request field `return_spec_tokens_details: true`): spec_verify_ct = engine
   steps of this request, spec_num_proposed_drafts / spec_num_correct_drafts = draft / accepted tokens;
   `usage.step_stats` (the per-request field proposed for gr8 in docs/gr8-step-counters-proposal.md).
2. Prometheus `/metrics` scraped before and after a request (c1) or a block of requests (c>1); the delta is
   attributed to that request or block. vLLM and gr8 export vLLM's names; SGLang its own `sglang:` names.
3. stream events (one SSE event per engine step): only where the server really sends one event per step.

What each engine exposes (read from the engines' sources, see README "Engine counters"):
- gr8 (S2 serve, g6front-rs/src/metrics.rs + frontend/g6s2shim.py): per-request rounds, draft tokens and
  accepted tokens are added to the vllm:spec_decode_* counters when the request ENDS (the engine's "m" line);
  vllm:request_decode_time_seconds_sum (first token -> end, measured by the shim) likewise at the end. No
  per-request field in the response, no engine-wide (batched) step counter, no per-step timing.
- vLLM v1: vllm:spec_decode_* counters per step (num_drafts counts a request's steps that had drafts),
  vllm:iteration_tokens_total histogram (count = iterations that produced outputs; outputless chunked-prefill
  iterations are not counted, so it is a lower bound on engine steps), request decode/prefill time histograms,
  TTFT and inter-token latency.
- SGLang: per-request sglext.spec_tokens_details (above); /metrics generation/prompt/cached token counters and
  TTFT/ITL histograms; spec acceptance only as a gauge (useless for deltas).
"""
import re

from . import client

# canonical name -> metric names that carry it (first present wins); all are counters (or histogram _sum/_count)
COUNTERS = {
    "spec_steps": ["vllm:spec_decode_num_drafts_total"],
    "draft_tokens": ["vllm:spec_decode_num_draft_tokens_total"],
    "accepted": ["vllm:spec_decode_num_accepted_tokens_total"],
    "iterations": ["vllm:iteration_tokens_total_count"],
    "iteration_tokens": ["vllm:iteration_tokens_total_sum"],
    "gen_tokens": ["vllm:generation_tokens_total", "sglang:generation_tokens_total"],
    "prompt_tokens": ["vllm:prompt_tokens_total", "sglang:prompt_tokens_total"],
    "cached_tokens": ["vllm:prompt_tokens_cached_total", "sglang:cached_tokens_total"],
    "prefix_queries": ["vllm:prefix_cache_queries_total"],
    "prefix_hits": ["vllm:prefix_cache_hits_total"],
    "decode_s": ["vllm:request_decode_time_seconds_sum"],
    "prefill_s": ["vllm:request_prefill_time_seconds_sum"],
    "finished": ["vllm:request_generation_tokens_count", "vllm:request_decode_time_seconds_count",
                 "sglang:num_requests_total"],
    "ttft_s": ["vllm:time_to_first_token_seconds_sum", "sglang:time_to_first_token_seconds_sum"],
    "ttft_n": ["vllm:time_to_first_token_seconds_count", "sglang:time_to_first_token_seconds_count"],
    "itl_s": ["vllm:inter_token_latency_seconds_sum", "sglang:inter_token_latency_seconds_sum"],
    "itl_n": ["vllm:inter_token_latency_seconds_count", "sglang:inter_token_latency_seconds_count"],
    # proposed for gr8 (docs/gr8-step-counters-proposal.md): engine-wide batched decode rounds
    "engine_steps": ["g6:s2_rounds_total", "g6:s2_steps_total"],
    "engine_step_s": ["g6:s2_round_seconds_total", "g6:s2_step_seconds_total"],
    "engine_step_sessions": ["g6:s2_round_sessions_total", "g6:s2_step_sessions_total"],
}
GAUGES = {"running": ["vllm:num_requests_running", "sglang:num_running_reqs"],
          "sglang_accept_length": ["sglang:spec_accept_length"]}
# accumulate only at request end on gr8 (a scrape during a request does not see it yet)
AT_END = {"gr8": ("spec_steps", "draft_tokens", "accepted", "decode_s", "prefill_s", "finished")}

_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_prometheus(text):
    """{metric name: value summed over label sets}; histogram buckets are skipped"""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m or m.group(1).endswith("_bucket"):
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def canonical(raw):
    out = {}
    for table in (COUNTERS, GAUGES):
        for k, names in table.items():
            for n in names:
                if n in raw:
                    out[k] = raw[n]
                    break
    return out


def scrape(url, headers=None, timeout=5):
    """canonical counters from url/metrics, or None (no endpoint, error, or nothing we know)"""
    try:
        text = client.http_text(url.rstrip("/") + "/metrics", timeout=timeout, headers=headers)
    except Exception:  # noqa: BLE001 - metrics are optional
        return None
    return canonical(parse_prometheus(text)) or None


def delta(before, after):
    if not before or not after:
        return None
    out = {}
    for k, v in after.items():
        if k in COUNTERS and k in before:
            out[k] = v - before[k]
        elif k in GAUGES:
            out[k] = v
    return out or None


def detect(meta, counters=None):
    """engine name from /v1/models owned_by, the metric families, or the Server header"""
    try:
        owner = (meta.get("models") or {}).get("data")[0].get("owned_by") or ""
    except (AttributeError, IndexError, TypeError):
        owner = ""
    owner = owner.lower()
    for name in ("gr8", "vllm", "sglang", "llama"):
        if name in owner:
            return "llama.cpp" if name == "llama" else name
    srv = (meta.get("server_header") or "").lower()
    if "llama" in srv:
        return "llama.cpp"
    if counters:
        if any(k.startswith("engine_") for k in counters):
            return "gr8"
        if "iterations" in counters:
            return "vllm"
        if "sglang_accept_length" in counters:
            return "sglang"
    return "unknown"


def engine_version(meta):
    """best-effort version string from /version (vLLM), /server_info or /get_model_info (SGLang)"""
    for path in ("/version", "/server_info", "/get_server_info"):
        v = meta.get(path)
        if isinstance(v, dict):
            for k in ("version", "sglang_version", "vllm_version"):
                if v.get(k):
                    return str(v[k])
    return None


def per_request(extras):
    """canonical per-request counters from response fields (SGLang sglext, proposed gr8 usage.step_stats)"""
    if not extras:
        return None
    st = extras.get("step_stats")
    if isinstance(st, dict) and st.get("steps"):
        out = {"spec_steps": st["steps"], "source": "response"}
        for a, b in (("draft_tokens", "draft_tokens"), ("accepted", "accepted_tokens"), ("decode_s", "decode_s"),
                     ("prefill_s", "prefill_s")):
            if st.get(b) is not None:
                out[a] = st[b]
        return out
    det = (extras.get("sglext") or {}).get("spec_tokens_details") if isinstance(extras.get("sglext"), dict) else None
    if isinstance(det, dict) and det.get("spec_verify_ct"):
        out = {"spec_steps": det["spec_verify_ct"], "source": "response"}
        if det.get("spec_num_proposed_drafts") is not None:
            out["draft_tokens"] = det["spec_num_proposed_drafts"]
        if det.get("spec_num_correct_drafts") is not None:
            out["accepted"] = det["spec_num_correct_drafts"]
        return out
    return None


def steps_of(spec):
    """engine decode steps of one request (c1) from a canonical delta or per-request fields: the speculative
    verify rounds. Without a drafter every step makes one token and the decomposition is trivial (None)."""
    if not spec:
        return None
    return spec.get("spec_steps") or None
