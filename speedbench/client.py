"""Streaming OpenAI-compatible client: one request, timed at the client.

Only the Python standard library. Handles the stream shapes of vLLM, SGLang, llama.cpp and gr8: deltas with
`content`, `reasoning_content` or `reasoning`, tool-call deltas keyed by index, usage-only chunks, chunks with
empty `choices`, and an optional cumulative `usage` on every chunk (vLLM `continuous_usage_stats`).
"""
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request

REASONING_KEYS = ("reasoning_content", "reasoning")


class RequestFailed(Exception):
    pass


def http_json(url, body=None, timeout=30, headers=None):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null"), dict(r.headers)


def http_text(url, timeout=10, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def _iter_sse(resp):
    """Yield the data payload of each SSE event (handles multi-line `data:` fields and CRLF)."""
    buf = []
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if buf:
                yield "\n".join(buf)
                buf = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip(" "))
    if buf:
        yield "\n".join(buf)


def stream_chat(base_url, body, timeout=900, headers=None, clock=time.perf_counter):
    """Send one streaming chat request; return client-side timings, token counts and the output.

    t_first / t_last are the arrival times of the first and last event that carried generated text (content,
    reasoning or a tool-call fragment). Token counts come from the server's `usage` only. A request counts as
    complete only if the stream ended normally with a finish_reason and a final usage; anything else raises
    RequestFailed. When every text event carried a cumulative usage (continuous usage stats) the decode
    interval's token count is exact: tokens(last text event) - tokens(first text event).
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    S = {"t_first": None, "t_last": None, "usage": None, "finish": None, "content": [], "reasoning": [],
         "tools": {}, "chunks": [], "n_events": 0, "ev_after_first": 0, "n_steps": 0, "extras": {}}
    t_send = clock()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise RequestFailed("HTTP %d: %s" % (e.code, detail)) from None
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise RequestFailed("connection: %r" % (e,)) from None
    done = False
    try:
        with resp:
            for data in _iter_sse(resp):
                if data.strip() == "[DONE]":
                    done = True
                    break
                _event(S, data, clock(), t_send)
    except RequestFailed:
        raise
    except (OSError, http.client.HTTPException, ValueError) as e:     # timeouts, resets, truncated bodies
        raise RequestFailed("stream interrupted: %r" % (e,)) from None
    t_end = clock()
    usage = S["usage"]
    if usage is None or not usage.get("completion_tokens"):
        raise RequestFailed("stream ended without usage.completion_tokens (no throughput can be computed)")
    if S["finish"] is None:
        raise RequestFailed("stream ended without a finish_reason (truncated response)")
    chunks = S["chunks"]
    exact = bool(chunks) and chunks[0][2] >= 0 and chunks[-1][2] >= 0
    out_c, out_r = "".join(S["content"]), "".join(S["reasoning"])
    tools = S["tools"]
    tool_list = [{"name": tools[k]["name"], "arguments": tools[k]["arguments"]} for k in sorted(tools)]
    h_text = hashlib.sha256(json.dumps([out_r, out_c, tool_list], ensure_ascii=False).encode()).hexdigest()
    details = usage.get("completion_tokens_details") or {}
    pdetails = usage.get("prompt_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _reasoning_tokens_from_chunks(chunks)
    t_first, t_last = S["t_first"], S["t_last"]
    return {
        "t_send": t_send, "ttft_s": None if t_first is None else t_first - t_send,
        "decode_s": None if t_first is None else t_last - t_first, "total_s": t_end - t_send,
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage["completion_tokens"],
        "cached_tokens": pdetails.get("cached_tokens"), "reasoning_tokens": reasoning_tokens,
        "boundary": "exact" if exact else "assumed",
        "first_chunk_tokens": chunks[0][2] if exact else None,
        "last_chunk_tokens": chunks[-1][2] if exact else None,
        "finish_reason": S["finish"], "done_marker": done,
        "n_chunks": len(chunks), "n_events": S["n_events"], "n_steps": S["n_steps"],
        "reasoning_chars": len(out_r), "content_chars": len(out_c), "tool_calls": len(tool_list),
        "sha256": h_text, "text": {"reasoning": out_r, "content": out_c, "tool_calls": tool_list},
        "chunks": chunks, "extras": S["extras"] or None,
    }


def _event(S, data, now, t_send):
    try:
        j = json.loads(data)
    except ValueError:
        raise RequestFailed("bad SSE payload: %r" % data[:200]) from None
    S["n_events"] += 1
    if j.get("error"):
        raise RequestFailed("stream error: %s" % (j["error"],))
    if j.get("usage"):
        S["usage"] = j["usage"]
        if isinstance(j["usage"], dict) and j["usage"].get("step_stats"):
            S["extras"]["step_stats"] = j["usage"]["step_stats"]     # proposed gr8 per-request counters
    if j.get("sglext"):
        S["extras"]["sglext"] = j["sglext"]                           # SGLang per-request spec details
    if j.get("choices") and S["t_first"] is not None:
        S["ev_after_first"] += 1
    got, chan = 0, ""
    for ch in j.get("choices") or []:
        d = ch.get("delta") or {}
        c = d.get("content")
        if c:
            S["content"].append(c)
            got += len(c)
            chan += "c"
        for k in REASONING_KEYS:
            rc = d.get(k)
            if rc:
                S["reasoning"].append(rc)
                got += len(rc)
                chan += "r"
                break
        for tc in d.get("tool_calls") or []:
            slot = S["tools"].setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["arguments"] += fn["arguments"]
                got += len(fn["arguments"])
            chan += "t"
        if ch.get("finish_reason"):
            S["finish"] = ch["finish_reason"]
    if got or "t" in chan:
        cum = (j.get("usage") or {}).get("completion_tokens")
        cum = -1 if cum is None else cum
        if S["t_first"] is None:
            S["t_first"] = now
        S["t_last"] = now
        S["n_steps"] = S["ev_after_first"]
        S["chunks"].append([round((now - t_send) * 1000, 2), got, cum, chan])


def _reasoning_tokens_from_chunks(chunks):
    """Exact reasoning-token count when every text chunk carried a cumulative usage (continuous usage)."""
    if not chunks or any(c[2] < 0 for c in chunks):
        return None
    prev, n = 0, 0
    for _, _, cum, chan in chunks:
        if chan == "r":
            n += cum - prev
        elif "r" in chan:      # a chunk that crosses the reasoning/content boundary: ambiguous
            return None
        prev = cum
    return n
