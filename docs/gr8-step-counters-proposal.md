# Proposal: per-request and per-step counters in gr8 for the speed decomposition

Status: proposal for an upstream PR to gr8 (kindlingai/gr8). Nothing here is implemented in gr8; speedbench
already reads these fields when they exist and falls back to what gr8 exposes today. Checked against gr8
`origin/main` at 3bdd2cb (2026-10-18), after PR #93 ("Report measured S2 request timing, KV capacity, and prefix
reuse on /metrics").

## What gr8 exposes today (S2 serve)

| need | gr8 today | where |
|---|---|---|
| engine steps of one request | yes, but only through `/metrics` and only when the request ENDS: `vllm:spec_decode_num_drafts_total` += the request's verify rounds | engine `m <rounds> <drafted> <accepted> <per-pos...>` line before `end` (engine/runtime/s2/sessions.h `terminal`), added up in frontend/g6s2shim.py `line()`, rendered by g6front-rs/src/metrics.rs |
| draft and accepted tokens of one request | yes, the same way (`_num_draft_tokens_total`, `_num_accepted_tokens_total`, `_per_pos_total`) | same |
| server-side decode time of one request | yes, at the end: `vllm:request_decode_time_seconds_sum` (shim clock, first tokens line to `end`) | g6s2shim.py `account()` (PR #93) |
| TTFT, ITL, prefill time, cached prompt tokens | yes (sums and counts, PR #93) | same |
| per-request counters in the HTTP response | **no**: `usage` has prompt/completion/cached/reasoning tokens only | g6front-rs/src/api.rs `usage()` |
| engine-wide (batched) decode steps, step time, sessions per step | **no** | — (`G6_S2_TIMELINE` prints means to stderr, off by default) |
| one SSE event per engine step | **no**: the front end drops events without a parsed delta | g6front-rs |

Consequences for speedbench:

- At c1, the `/metrics` scrape before and after each request is exact: the request's verify rounds, draft and
  accepted tokens, and its server-side decode time all land before the client sees `[DONE]` (the shim adds the
  `m` line and the timing before it sends the finish frame). `run --metrics` therefore gives the full
  decomposition on gr8 today: tokens/step = (completion_tokens - 1) / rounds, step time (client and server),
  acceptance, drafts per step (which varies with `G6_MTP_KSTOP`).
- At c>1 the counters can only be attributed to a block of requests, not to a request: `load --metrics` reports
  tokens per stream-step and the server-side step time of a stream (sum of decode time / sum of rounds), but
  not the engine's batched step time or how many sessions shared a step, because gr8 counts rounds per session
  and has no engine-wide step counter.

## The smallest addition

Two independent parts; the first needs no engine change.

### P1. Per-request counters in the response (shim + front end, no engine change)

The shim already holds every request's `m` line (`Req.m`) and its timing (`t_admit`, `t_first`, end). Pass them
on in the `finish` frame and let the front end put them in the final usage object when the request asks for it:

- PROTOCOL.md, `finish` frame: optional `spec: {"rounds": R, "draft_tokens": D, "accepted": A,
  "accepted_per_pos": [...]}` and `timing: {"prefill_s": p, "decode_s": d}`. Optional fields, absent when unknown
  (the same rule as the `metrics` frame).
- g6s2shim.py `finish()`: add them from `r.m` and the request's clocks (about 10 lines).
- g6front-rs api.rs: when the request body has `"return_step_stats": true` and the stream has `include_usage`,
  add to the final `usage` object
  `"step_stats": {"steps": R, "draft_tokens": D, "accepted_tokens": A, "accepted_per_pos": [...], "prefill_s": p, "decode_s": d}`
  (and the same in a non-streaming response). Opt-in, so strict OpenAI clients see no new field.
- Tests: one shim test (finish frame carries the fields), one front-end test (usage has `step_stats` only when
  asked).

What it buys: exact per-request decomposition at any concurrency, without scraping `/metrics` around every
request (two scrapes per request cost a little engine time on a 4-rank engine). `decode_s` has the same
boundaries as the rounds (first tokens line to the end), so `decode_s / steps` is the request's step time; the
client's decode interval starts at the first *visible* event and does not match the step count exactly.

speedbench side (implemented): `run --per-request-stats on` sends `return_step_stats: true` and reads
`usage.step_stats` (engines.per_request); `--per-request-stats auto` does this only for SGLang until gr8 has it.

### P2. Engine-wide round counters (engine, small)

On the head rank, in the S2 batch (engine/runtime/s2_batch.h, where a round's verify step, its drafting steps and
the per-session accounting `++stats_[q].rounds` happen), keep engine-wide totals:

- `rounds`: decode rounds executed (one per batched round, any number of sessions);
- `round_sessions`: sum over rounds of the sessions in the round;
- `round_us`: decode time on rank 0, accumulated as closed intervals of a steady clock. An interval opens when a
  round starts and closes at the first of: the start of the next round (the drafting steps of the round, also on
  the side stream, and the host gap after it are inside), the start of a prefill quantum, the moment no session is
  decoding (idle), or a terminal publication (below). A closed interval is added exactly once; after a prefill
  quantum or idle time the next round opens a new interval, so neither is counted;
- `prefill_us`: wall time of prefill quanta (closed the same way).

Publication closes the open interval first: before writing a request's `m` line, the decode thread adds the time
from the open interval's start to now to `round_us` and restarts the interval at now (so the rest of the round
is counted in the next publication, not twice). The totals in the `m` line therefore cover every round up to that
instant.

Publication with a barrier: append the four totals to every request's `m` line (written by the decode thread when
the request ends, before its `end` line), e.g. `m <rounds> <drafted> <accepted> <acc_pos...> | <rounds_total>
<round_sessions_total> <round_us_total> <prefill_us_total>`. The shim keeps the latest totals it has read and sends
them in the `metrics` frame (`engine: {...}`); metrics.rs renders `g6:s2_rounds_total`,
`g6:s2_round_sessions_total`, `g6:s2_round_seconds_total`, `g6:s2_prefill_seconds_total`. When the last request of
a benchmark block ends, the totals include every round up to that moment. (Writing them into `status.s2` is not
enough: the control thread publishes it from a 1-ms polling loop, `service.h controller()`, so a request's end is
not a publication barrier there.)

What it buys, for a block of requests at c>1 with wall time W:

    aggregate decode tok/s = (tokens per session-round) x (sessions per round) / (seconds per round) x (round time / W)

with tokens per session-round = 1 + accepted / per-session rounds (already exported), sessions per round =
round_sessions / rounds, seconds per round = round_us / rounds, and the decode share of the wall time =
round_us / W (the rest is prefill and idle). Each factor is a separate lever: acceptance, batching, step cost,
scheduling.

speedbench reads these names when present (engines.COUNTERS `engine_steps`, `engine_step_s`,
`engine_step_sessions`; the mock server's `--flavor gr8 --proposed` imitates them); the names will follow whatever
the upstream PR settles on.

## Not proposed

- Per-step timing histograms: the means above are enough for A/B; the timeline knob covers deep dives.
- Emitting one SSE event per engine step: changes the stream for every client; the counters are cheaper.
