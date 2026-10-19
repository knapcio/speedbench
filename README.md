# speedbench

A paired A/B speed benchmark for LLM serving endpoints that speak the OpenAI chat-completions API
([vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang),
[llama.cpp](https://github.com/ggml-org/llama.cpp) server, [gr8](https://github.com/kindlingai/gr8), ...). It measures
decode speed, prefill and time to first token (TTFT), concurrency, long contexts, multi-turn agent sessions served
from the prefix cache, and optionally energy per token, and it answers one question with an honest error bar:
**is configuration B faster than A, and by how much?** For decode it also explains the answer:
**tok/s = tokens per step x steps per second**, from the engine's own counters. An optional quality guard checks
that a speed lever did not break the outputs.

Python 3 standard library only; run the commands from the repository root. How it was designed and reviewed:
[DESIGN.md](DESIGN.md).

```
# A/B: both arms interleaved in ONE session (the only design that resolves ~1 % effects; see "Sessions")
python3 -m speedbench run --arm A=http://127.0.0.1:8100 --arm B=http://127.0.0.1:8100,extra='{"...": 1}' \
        --metrics --out ab-                                                      # decode, ~19 min (2 x full)
python3 -m speedbench compare ab-A.json ab-B.json

python3 -m speedbench run --arm A=URL --arm B=URL,extra=... --mode agent --out ag-       # multi-turn agent, ~10 min
python3 -m speedbench run ... --quality-guard                                    # + exact-answer guard, T=0
python3 -m speedbench run ... --power-cmd "nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits -lms 200"
python3 -m speedbench run ... --temperature 0.7 --seeds 1,2,3                    # T>0, paired per (prompt, seed)

python3 -m speedbench run --url URL --label base --metrics --out base.json     # one arm, full: ~9-10 min
python3 -m speedbench load --url URL --conc 1,2,4,8 --metrics --mix agentic     # concurrency + agentic mix: ~10 min
python3 -m speedbench longctx --url URL --metrics                               # TTFT 8K..128K, decode 32/64/128K: ~12 min
python3 -m speedbench report base.json                                          # per-category decomposition
python3 -m speedbench registry table --readme README.md                         # results table below
```

| command | what | runtime per arm (~40 tok/s decode, ~1.3K tok/s prefill) | use |
|---|---|---|---|
| `run --mode full`, two arms interleaved | 73 prompts x 256 tokens, c1 | ~9-10 min (+~6 s warm-up prefill) | **publishable** A/B verdict on decode tok/s (with zero dropped prompts); with `--preregister step` also on step time |
| `run --mode quick` | fixed 30-prompt subset x 160 tokens | ~2.5-3 min | **screening** (verdicts labelled SCREEN ONLY) |
| `run --mode agent`, two arms interleaved | 8 conversations, 48 requests x 160 tokens, prefix cache | ~4.5-5 min (mock: 4.8 min) | TTFT of cached turns and turn latency: **publishable** with zero dropped turns, for effects consistent across conversations (history frozen, no luck; CI over 8 conversations); decode tok/s: indicative |
| `run --mode agent-quick` | 4 conversations x 4 turns | ~1.5-2 min (mock: 1.8 min) | **screening** |
| `run ... --temperature T --seeds 1,2,3` | every prompt once per seed | 3x the mode | **publishable** when interleaved and no (prompt, seed) pair was dropped; otherwise NOT PUBLISHABLE (descriptive) or WITHHELD; `compare` prints the MDE for more seeds |
| `run ... --metrics` | + engine counters around every request (decomposition, step-time rule) | +2 scrapes per request (~1 %) | |
| `run ... --quality-guard` | + 30 exact-answer items at T=0, text kept for the divergence summary | ~2-3 min (up to ~6.5 min if every item thinks for 512 tokens) | a **guard** against gross breakage, not a quality benchmark |
| `--power-cmd` (`run`, `load`) | + power sampling: J/token, tok/J | +5 s idle reading | descriptive; paired J/token CI in `compare` |
| `load` | c in {1,2,4,8} x 60 s, closed loop; `--mix agentic` adds 180 s of the agentic mix | ~5 min; ~9-10 min with the mix | **descriptive** (no A/B verdict) |
| `longctx` | TTFT cold/cached at 8K..128K, decode at 32K/64K/128K, 2 documents per size | ~12-15 min (prefill-bound) | **descriptive** |
| `compare`, `report`, `registry` | analysis only | seconds | |

"Publishable" means a paired verdict with a confidence interval whose assumptions are checked in the report
(interleaved arms, no withheld workload, sentinels) and **zero outcome-dependent drops**: if any prompt, turn or
(prompt, seed) pair was dropped because an output failed, was too short or could not be timed, the verdict is
labelled NOT PUBLISHABLE (descriptive only; the numbers and the drop count are still printed), in every mode; "screening" means a quick filter for candidates that must be
confirmed with a publishable design. Runtimes are estimates from published TP4 numbers of GLM-5.3 on gr8 on 4x DGX
Spark, except the agent modes, which were also measured on the mock server tuned to that speed (34 tok/s decode,
1.3K tok/s prefill); every result file records its own `elapsed_s`. The warm-up of every mode includes one cold
8K-token prefill (~6 s; `--warmup-prefill 0` removes it).

## Why another benchmark

With speculative decoding, decode speed depends on how well the drafter predicts the text, and the text depends
on the engine's numerics. Any change that alters the T=0 output (a different floating-point order, other drafts
and therefore other verify widths) puts the generation on a different trajectory with different draft
acceptance. That shifts tok/s by several per cent by luck, independently of the change's real cost.

A benchmark that sends **one prompt several times at T=0** ([RigMark](https://github.com/alexellis/rigmark)'s decode workloads) repeats the *same*
trajectory: the repeats average out timing jitter but never trajectory luck. Its run-to-run spread looks tiny,
so a lucky or unlucky trajectory reads as a real effect. Example from our fleet: levers A+B+C +3.6 %, A alone
+0.5 %, B+C -2.1 % on RigMark prose (not additive), while a 24-prompt set gave +3.1 % for A+B+C.

We measured the luck on our own data (36 configurations, 99 decset runs of 24 prompts, 384 tokens,
GLM-5.3 on gr8, 4x DGX Spark; `calibration/decset-runs-2026-10.json`), and the session effect on 2026-10-19:

| source of variation | value |
|---|---|
| trajectory-changing lever (output text differs): per-prompt SD of the paired log ratio | **4.3 %** (heavy tails; single prompts to -18 %) |
| trajectory-preserving lever (bit-exact output): per-prompt SD | 0.2-0.6 % |
| same configuration, repeated run in one session (request noise) | ~0.4 % per prompt; interleaved pairs agree to **~0.1 %** overall |
| same configuration, bit-exact, **two sessions a few hours apart** | **~2 %** (decset prose 43.75-44.21 vs 44.82 / 44.82) |

The self-test replays that luck: a RigMark-style comparison (one prompt x5, repeats treated as samples)
declares a "significant" difference **about 80 % of the time when there is none**. Averaging over many different
prompts is the only cure for luck, and interleaving both arms in one session is the only cure for session drift.

## Method

1. **Many prompts, paired.** 73 original prompts in 6 categories (prose 18, code 14, structured 12,
   agentic/tool-call 12, multilingual 8 (de, es, fr, it, pt, uk, zh, ja), long-context 9), in a fixed interleaved
   order, same in every run. For each prompt *i* the statistic is d_i = log(tok/s of B) - log(tok/s of A); the
   effect is the mean of d_i (a geometric-mean speed ratio), equal weight per prompt.
2. **Interleaved sessions.** Both arms run in one session (`--arm A=... --arm B=...`), alternating A,B / B,A per
   job, so the session shock cancels and only a small residual remains (prior 0.1 %, estimated from the data
   when there are >= 2 interleaved sessions). See "Sessions".
3. **Honest error bar.** Variance = stratified variance of d_i over prompts (contains trajectory luck and request
   noise) + a session term. At T=0 a configuration writes the same text on every boot, so the per-prompt luck
   is fixed per configuration (and per `--variants` tag): extra sessions or boots average timing and session
   noise, never luck; they are not counted as extra luck samples (the prompt term is divided by the number of
   prompts only). 95 % CI and p-value from Student t with Welch-Satterthwaite degrees of freedom. One verdict for
   the whole set: **B FASTER**, **B SLOWER**, or **NO DETECTABLE DIFFERENCE** with the minimum effect the run
   could have detected (MDE at 80 % power). Categories get estimates and CIs, but no verdicts.
4. **Headline = end-to-end decode tok/s and TTFT.** Per request: tok/s = tokens generated between the first
   and the last streamed event that carried text (content, reasoning or tool-call) / the time between those
   two events. When the server sends per-chunk usage (`stream_options.continuous_usage_stats`, probed
   automatically) the token count is exact: cumulative tokens at the last text event minus those at the first.
   Without it the count is completion_tokens - 1. `compare` refuses to mix the two. A request counts only if the
   stream ended with a finish_reason and a usage block; anything else is a failed request, never a measurement.
5. **Decomposition (why).** With `--metrics`, per request: engine steps (verify rounds), tokens per step =
   (completion_tokens - 1) / steps, step time = decode interval / steps inside it, draft acceptance, drafts per
   step, and the engine's own (server-side) step time. log tok/s = log tokens/step - log step time, exactly. A
   kernel lever moves step time; a drafter lever moves tokens/step. See "Speed decomposition".
6. **Diagnostics.** OUTPUT IDENTITY (byte-identical outputs in A and B: no luck in play), sentinels (in-run
   drift and T=0 determinism), HORIZON (luck vs output length), sensitivity lines (median, Hodges-Lehmann,
   winsorized mean, prompt bootstrap, leave-one-out), largest per-prompt losses and gains.
7. **Controlled state.** Discarded warm-up that includes one long cold prefill (8K tokens; the first long
   prefill after a boot runs on cold kernels and graphs and must never be timed); a `max_tokens=1` priming
   request before every measured one so both arms decode in the same prefix-cache state; server/model metadata,
   settings, the effective request, the prompt-set hash and the operator's host metadata are stored.
   `--extra-body` may not touch benchmark-controlled fields. `compare` refuses runs with different settings or
   effective requests, the same run given twice, interrupted runs, and mixed token boundaries; prompts that fail
   in any run are dropped from both arms as a whole, category weights stay frozen, and the verdict is withheld if
   more than max(2, 5 %) of the prompts drop out.

## Sessions: interleave, never compare across sessions

The same bit-exact configuration measured ~2 % apart in two sessions a few hours apart, while interleaved pairs
inside one session agree to ~0.1 %. A session-level shift of that size is larger than most levers. Therefore:

- **Interleave both arms in one session** whenever the lever can be switched without a restart (an env flag
  read per request, a model alias, an extra body field, two servers side by side): `run --arm A=URL
  --arm B=URL,extra={...}`. `compare` recognises the shared session and checks that the arms really were
  interleaved (mean request time of A and B within 5 % of the session); otherwise it prints **NOT INTERLEAVED**
  and treats the arms like separate sessions (session prior, prior-conditional verdict).
- **Levers that need a restart:** alternate sessions A B A B ... (at least 3 per arm), label each run with its
  `--block`, and compare all of them (`compare --a A1 A2 A3 --b B1 B2 B3`). The session SD is then estimated from
  the data (two-way ANOVA, pooled with the prior); with one session per arm it is only assumed.
- **Several interleaved sessions** (`--a A_s1 A_s2 --b B_s1 B_s2`, same session ids pairwise): the residual of the
  paired difference from session to session is estimated by ANOVA on the per-session differences, pooled with the
  0.1 % prior as 4 pseudo-observations. With 2-3 sessions that interval is prior-conditional (the self-test: a true
  residual of 0.3 %, 3x the prior, gives ~24 % false positives with 2 sessions), so `compare` also prints the CI
  with the session term from the data alone and warns when the estimate exceeds twice the prior.
- Results of arms from different sessions are labelled **CROSS-SESSION** in the verdict, with a banner. The
  default session prior for that case is 1.0 % per arm (`--boot-prior`; ~1.4 % between two sessions, matching the
  ~2 % observed shift in size), and the CI under a 2 % session SD is printed next to it. The previous default,
  0.25 %, was the boot-to-boot value measured inside one decset session and is too small across sessions.

## Host preparation

Results depend on host state; record it with the metadata flags (stored in the result and the registry):

- **Compact memory before starting the server** on unified-memory systems (DGX Spark / GB10): a fragmented host
  made one rank 20-80 % slower on our fleet. `sync; echo 1 | sudo tee /proc/sys/vm/compact_memory` on every node,
  then start the server. Record `--host-compacted yes|no` and `--fresh-boot yes|no`.
- **GPU clock policy** changes speed: record it, e.g. `--clock-policy "cap 2200 MHz"`. Never compare runs with
  different clock policies; `compare` prints a note when operator metadata differs between runs.
- **Warm-up:** every mode runs a discarded warm-up with a long cold prefill before timing anything (`run`: 8K
  tokens; `load --mix`: the long document; `longctx`: the largest context). `--warmup-prefill 0` disables it.
- Nothing else on the fleet; same image, same model files; note the engine commit (`--engine-commit`) and the
  configuration (`--config`). Run the client next to the server (SSE framing and network are in the timing).

## Speed decomposition

`tok/s = tokens per step x steps per second`. With speculative decoding a trajectory-changing lever moves
tokens/step by luck (+-4-6 % per prompt), while the step time hardly moves with the text (~0.5-1 % per prompt).
So a lever that only changes the cost of a step (kernels, communication, scheduling) is visible in the step time
long before tok/s can resolve it.

### Engine counters

| engine | steps of one request | draft / accepted tokens | server-side decode time | at c>1 |
|---|---|---|---|---|
| **gr8** (S2 serve, after PR #93) | `/metrics` `vllm:spec_decode_num_drafts_total` delta = the request's verify rounds, **added when the request ends** | `_num_draft_tokens_total`, `_num_accepted_tokens_total` (also at the end) | `vllm:request_decode_time_seconds_sum` (shim clock, at the end) | per block only: rounds are per session; no engine-wide step counter |
| **vLLM** v1 | `vllm:spec_decode_num_drafts_total` delta (every step that had drafts) | same names | `vllm:request_decode_time_seconds_sum` | per block; `vllm:iteration_tokens_total_count` counts iterations that produced outputs (a lower bound on engine steps: outputless chunked-prefill iterations are not counted) |
| **SGLang** | per request in the response: `sglext.spec_tokens_details.spec_verify_ct` (sent `return_spec_tokens_details: true`) | `spec_num_proposed_drafts`, `spec_num_correct_drafts` | not per request | per request (exact at any concurrency) |

gr8 has **no per-request counters in the response and no engine-wide (batched) step counter or step time**.
At c1 that does not matter: the scrape before and after each request is exact (the counters land before the
client sees `[DONE]`), so `run --metrics` gives the full decomposition on gr8 today. At c>1 only block totals can
be attributed. The smallest upstream addition that closes the gap is in
[docs/gr8-step-counters-proposal.md](docs/gr8-step-counters-proposal.md): (P1, shim + front end only) the
request's rounds, draft/accepted tokens and decode time in `usage.step_stats` on request; (P2, small engine
change) engine-wide round, round-session, round-time and prefill-time totals, published through each request's
`m` line so that a request's end is a barrier. speedbench already reads both (`--per-request-stats on`; the mock
imitates them with `--flavor gr8 --proposed`).

Stream events are engine steps only if the server sends one event per step (not on gr8, whose front end drops
events without a parsed delta); without counters the report says so.

### What is reported

- `run --metrics`: per request `steps_engine`, `tokens_per_step`, `step_ms_server` (the engine's decode time of
  the request per step: same boundaries as the step count), `step_ms` (client decode interval per step, where the
  steps inside the client interval are inferred from tokens, because the first visible event can come several
  steps after the first token: an approximation, shown but never used for a decision), `accept_rate`,
  `draft_per_step`. A `/metrics` delta during which a number of requests other than one finished is discarded
  (`spec_note`).
- `report RUN.json`: per category, the geometric mean and 95 % CI over prompts of tok/s, step time, tokens/step,
  server step time, acceptance and drafts per step.
- `compare`: paired B vs A for each of them, overall and per category, with CIs and MDEs (DECOMPOSITION block).
- `load --metrics`: per concurrency level, tokens per stream-step, stream step time (server), and where the
  engine counts them, engine steps and batch occupancy.

### The step-time rule (pre-registered)

Declare the rule **before** the run: `run ... --preregister step` (recorded in the result). It is meant for
step-cost levers: kernels, communication, scheduling; never for drafter, acceptance or verify-width levers.
`compare` then shows the STEP-TIME RULE block, and the step-time verdict **decides** only if all checks pass:

1. declared before the run, with its margin (`run --preregister step --tps-margin 2`; one recorded margin in
   the B runs); `compare --decide-on step` afterwards is shown as post hoc and never decides;
2. the workload is valid (no WITHHELD verdict) and the arms ran interleaved in one session;
3. steps counted by the engine (response fields or `/metrics`) in every request of both arms, not stream events;
4. the engine's own step time is available in every request (`step_ms_server`; the client step time is an
   approximation and never decides; SGLang does not report a per-request decode time, so the rule cannot decide
   there);
5. acceptance unchanged within the margin: the 90 % CI of the tokens/step change lies inside +-2 % (TOST at 5 %);
6. draft length unchanged in the same sense, when the engine reports draft tokens (with `G6_MTP_KSTOP` the
   number of drafts per step varies; a lever that drafts more pays more per step and is judged on tok/s);
7. the tok/s verdict is not contradicted: a tok/s result significant in one direction can only be confirmed by
   the step rule, never overturned or turned into "no difference";
8. the tok/s point estimate (= step effect + tokens/step change) points the same way as the step verdict: a
   measured acceptance loss that outweighs the step gain voids the rule even when it is inside the margin.

The rule rests on the declared assumption that a step-cost lever does not change acceptance in expectation; the
equivalence check bounds a violation, it does not prove there is none. The block always prints the tok/s range
that remains when the measured tokens/step uncertainty is added to the step-time CI.

Otherwise the tok/s verdict decides; in quick mode the step decision is labelled SCREEN ONLY as well. The block also prints the tok/s effect implied by the step time at equal
acceptance, and the range that remains when the measured tokens/step uncertainty is added. Check 3 passes in
about 9 of 10 full runs when acceptance is truly unchanged (73 prompts, luck SD ~5 % at 256 tokens: 90 % CI
+-1.0 %); on the quick set (30 x 160) it passes only for (nearly) bit-exact levers. Step time still carries the
session term, so interleave.

## Sensitivity and runtime

From `python3 -m speedbench selftest --sims 300` (simulated experiments analysed by the same code; trajectory
luck resampled from our fleet data, scaled to the output length as sqrt(384/L)). Interleaved = both arms in one
session, residual 0.07 % per arm; cross-session = session SD 1 % per arm. MDE80 = smallest effect detected 80 % of
the time (~1.43 x the CI half-width). Step-time rows assume a per-prompt step-time SD of 0.6 % for a
trajectory-changing step-cost lever (an assumption until measured on the fleet).

| design | time at ~40-45 tok/s | false pos. | power +1 % | power +2 % | CI +- | MDE80 |
|---|---|---|---|---|---|---|
| **full** (73 x 256), trajectory-changing, interleaved | ~19 min (2 arms) | 4.3 % | 0.32 | 0.88 | 1.22 % | **~1.7 %** |
| full + `--variants 4`, trajectory-changing, interleaved | ~75 min | 3.0 % | 0.63 | 0.99 | 0.87 % | ~1.25 % |
| T>0, full, 1 seed per prompt, interleaved | ~19 min | 4.3 % | 0.35 | 0.85 | 1.24 % | ~1.8 % |
| T>0, full, 3 seeds per prompt, interleaved | ~57 min | 4.0 % | 0.74 | 1.00 | 0.73 % | ~1.05 % |
| T>0, quick 30, 3 seeds per prompt, interleaved | ~18 min | 5.0 % | 0.24 | 0.77 | 1.49 % | ~2.1 % |
| full, trajectory-preserving (bit-exact), interleaved | ~19 min | 5.7 % | 1.00 | 1.00 | 0.25 % | **~0.35 %** |
| full, trajectory-preserving, 2 interleaved sessions | ~38 min | 3.0 % | 1.00 | 1.00 | 0.20 % | ~0.3 % |
| **step time**, full, step-cost lever, interleaved (`--metrics`) | ~19 min | 3.7 % | 1.00 | 1.00 | 0.27 % | **~0.4 %** |
| step time, quick, step-cost lever, interleaved | ~6 min | 5.0 % | 1.00 | 1.00 | 0.36 % | ~0.5 % |
| **quick** (30 x 160), trajectory-changing, interleaved | ~6 min | 3.3 % | 0.15 | 0.33 | 2.53 % | ~3.6 % |
| cross-session, full, trajectory-changing, 1 session per arm | ~10 min per arm | 4.3 % | 0.08 | 0.19 | 3.02 % | ~4.3 % |
| cross-session, full, trajectory-preserving, 1 session per arm | ~10 min per arm | 5.3 % | 0.12 | 0.26 | 2.78 % | ~4.0 % |
| cross-session, trajectory-preserving, 3 sessions per arm (A B A B A B) | ~60 min | 4.3 % | 0.17 | 0.58 | 1.72 % | ~2.5 % |
| decset.py-like (24 x 384), trajectory-changing, interleaved | ~7 min | 5.0 % | 0.27 | 0.59 | 1.81 % | ~2.6 % |
| RigMark-like (1 prompt x5, ~1000 tok), repeats taken as samples | ~2 min per arm | **80 %** | — | — | (0.53 %, false) | honest: ~7 % |

Stress rows (`selftest`): true session SD 1.4 % with the 1 % prior -> coverage 0.82; in-run drift of one arm only
(1 % over the run) -> 0.90; 30 % nondeterministic T=0 requests -> 0.97; 2 interleaved sessions whose residual is
0.3 % (3x the prior) -> 0.76. The CI is honest when its assumptions hold; the sentinel, interleaving and session
warnings exist to tell you when they do not.

What this means:

- A **+1 % lever that changes the T=0 text cannot be confirmed in one full interleaved run** (MDE ~1.7 %). If it
  is a step-cost lever, the step-time rule resolves ~0.4 % in the same run. Otherwise it needs more
  trajectories: `--variants 4` (~1.25 %) or T>0 with several seeds.
- A **bit-exact lever** (OUTPUT IDENTITY 73/73) has no luck: interleaved, ~0.35 % is resolvable in one session.
- **Across sessions nothing below ~4 % is resolvable with one session per arm**, and ~2.5 % with three.
- **Quick mode is a smoke test** for tok/s (MDE ~3.6 %; `compare` labels it "SCREEN ONLY"), but its step time
  is precise.
- **At T>0 every seed is a new trajectory** (in both arms), so more seeds buy precision the way `--variants` does at
  T=0, and more cheaply in correlation terms (seeds are independent draws). The T>0 rows assume the per-trajectory
  luck equals the T=0 luck measured on our fleet.

## Running it

```
python3 -m speedbench run --url URL [--model ID] --label NAME --out FILE.json [options]
python3 -m speedbench run --arm A=URL[,model=M][,extra=JSON] --arm B=URL[...] --out PREFIX [options]
```

| option | default | meaning |
|---|---|---|
| `--mode full\|quick\|agent\|agent-quick` | full | 73 prompts x 256 tokens, the fixed 30-prompt subset x 160 tokens, or the multi-turn agent conversations (48 / 16 requests x 160 tokens) |
| `--isolate text\|field\|none` | text | agent modes: prefix-cache isolation between arms and runs (see "Multi-turn agent mode") |
| `--max-tokens N` | 256 / 160 | output cap (thinking included); `ignore_eos` is never sent |
| `--effort low` | low | reasoning effort; `''` sends none |
| `--effort-field template\|top\|both` | template | `chat_template_kwargs.reasoning_effort` (GLM on gr8/vLLM/SGLang) or OpenAI top-level `reasoning_effort` |
| `--temperature T --seeds 1,2,3` | 0 | T>0: every prompt once per seed (OpenAI `seed`), paired per (prompt, seed), averaged per prompt |
| `--variants N` | 1 | T=0: N deterministic trajectories per prompt via a tag "(Request tag: v)" on the last user message |
| `--repeats N` | 1 | re-send every job (adds trajectories only if the engine is nondeterministic) |
| `--arm L=URL[,model=M][,extra=JSON]` x2 | — | two endpoints in one session, alternating A,B / B,A per job (the A/B design) |
| `--metrics` | off | scrape `/metrics` around each request: speed decomposition |
| `--per-request-stats auto\|on\|off` | auto | ask for per-request step counters in the response (auto: SGLang with `--metrics`) |
| `--preregister tok\|step` | tok | the decision rule declared before the run (see "The step-time rule") |
| `--warmup N`, `--warmup-prefill T` | 3, 8192 | warm-up requests and the long cold warm-up prefill in tokens |
| `--extra-body JSON` | {} | merged into every request (engine-specific fields); benchmark-controlled fields are refused |
| `--block ID` | — | session label: runs of one arm with the same label are one session |
| `--no-prime` | prime on | skip the priming request (TTFT then includes prefill of an uncached prompt) |
| `--conc C` | 1 | C concurrent streams over the prompt set (descriptive; use `load` for concurrency) |
| `--skip-tools` | off | drop the 4 prompts that send a `tools` array (engines without tool-call support) |
| `--keep-text`, `--no-chunks` | off, on | store generated text; drop chunk traces |
| `--header 'K: V'`, `$OPENAI_API_KEY` | — | auth / extra headers |
| metadata | `unknown` | `--quant`, `--engine`, `--engine-version`, `--engine-commit`, `--config`, `--hardware`, `--thinking`, `--host-compacted`, `--fresh-boot`, `--clock-policy`, `--notes` |
| `--quality-guard`, `--guard-max-tokens N` | off, 512 | run the 30 exact-answer guard items at T=0 in every arm after the workload; keep the text |
| `--power-cmd CMD` (repeatable), `--power-kind gpu\|wall\|other`, `--power-idle S` | off, gpu, 5 | power sampling (see "Energy") |
| `--registry PATH` | — | append this run to the registry (e.g. `results/registry.jsonl`) |

All modes share the metadata flags, `--metrics`, `--effort`, `--extra-body`, `--header` and `--registry`.

### Recommended protocol

- **Screen an in-session lever:** `run --arm base=URL --arm lever=URL,extra={...} --metrics` (full), then
  `compare`. Start a new stack with an A-vs-A session (same config under two labels) to see its residual.
- **Kernel / communication lever:** the same, with `--preregister step` declared before the run.
- **Restart-only lever:** sessions A B A B A B with `--block s1..s6`; `compare --a ... --b ...`; prepare the host
  identically before each session (compaction, clocks).
- **Re-run** when: sentinel drift > ~0.5 %, NOT INTERLEAVED, failed requests, or a session-SD warning.
- Never use `--force` output as a verdict, and never compare across engines without checking the token
  boundary line (per-chunk usage on in both or off in both).

### gr8 / GLM-5.3 example

```
python3 -m speedbench run --arm base=http://127.0.0.1:8100 --arm l2=http://127.0.0.1:8100,extra='{"...": 1}' \
    --metrics --preregister step --out s1- --block s1 --host-compacted yes --fresh-boot yes \
    --clock-policy "cap 2200 MHz" --quant "gr8 EXLR8" --hardware "4x DGX Spark TP4" --thinking "on, effort low"
```

`--effort low` is sent as `chat_template_kwargs: {"reasoning_effort": "low"}` (thinking stays on). gr8 also
accepts the top-level field (`--effort-field top` or `both`); it takes precedence there.

### Engine notes

- **vLLM:** reasoning arrives as `reasoning` (newer) or `reasoning_content` (older); both are read.
  `continuous_usage_stats` gives exact per-chunk token counts. `/tokenize` is used for long-context sizing.
- **SGLang:** start with a reasoning parser so thinking is streamed as `reasoning_content`; tool prompts need a
  tool-call parser, otherwise use `--skip-tools`. With `--metrics` the run asks for `return_spec_tokens_details`
  and reads `sglext.spec_tokens_details` from the stream (the streaming placement is not yet verified on a real
  SGLang server; check the first result's `steps_source`).
- **gr8:** see "Engine counters". Check the deployed revision's per-chunk usage support on the first run (the
  result's `exact_boundary` count).
- **llama.cpp server:** needs `--jinja` for `tools` and for template kwargs; reasoning appears as
  `reasoning_content` with `--reasoning-format deepseek`. If usage is missing at the end of the stream the request
  fails loudly (throughput is never estimated from characters).
- If the server rejects `stream_options.continuous_usage_stats`, the run detects it on the probe request and
  continues without it.

## Concurrency and the agentic mix (`load`)

```
python3 -m speedbench load --url URL --conc 1,2,4,8 --duration 60 --metrics [--mix agentic --long-tokens 32768]
```

- **Levels:** c closed-loop streams, each sending distinct prompts from the set (no tool or long-context
  prompts; every stream starts at a different prompt, so no stream replays another's trajectory) back to back
  for `--duration` s. Reported per level: **aggregate decode tok/s** (tokens arriving between the moment every
  stream had its first token and the stop time, with a CI over 10-s bins; needs per-chunk usage, otherwise only
  `block_tok_s` = all completion tokens over the block's wall time is reported), **per-stream decode tok/s** (per
  request, geometric mean and CI), **TTFT** median/p90 (includes queueing), and with `--metrics` the block's
  tokens per stream-step, stream step time (server) and, where the engine counts them, batch occupancy.
- **Agentic mix** (`--mix agentic`, `--mix-duration 180`): one long session (a generated document of
  `--long-tokens`, first turn cold, later turns the same document with a new question, served from the prefix
  cache; `--long-max-tokens 512` per turn) while short requests arrive as a Poisson process (`--short-every 8` s,
  `--short-max-tokens 160`). Reported per class: TTFT and decode tok/s, and the long session's decode tok/s
  split by what the short requests were doing (absent / waiting or prefilling / decoding): the cost of an
  arrival to the long session. The time between two long-session events is split at every change of the short
  requests' state and its tokens are spread over the parts in proportion to time (only event times are observed).
- All numbers are descriptive (streams share the engine; requests are not independent).

## Long context (`longctx`)

```
python3 -m speedbench longctx --url URL [--contexts 8k,16k,32k,64k,128k] [--decode-contexts 32k,64k,128k] [--reps 2]
```

For every size and repetition: **cold** = a generated document with a unique salt on its first line (no prefix
cache can serve it) + a question; **cached** = the same document + another question (the cache serves the
document). Decode contexts decode `--decode-tokens 256`; the other sizes stop after 4 tokens. Documents are
deterministic (the prompt set's seeded generators: minutes, logs, code, CSV, chats, requirements, a chronicle,
configuration, diffs) and sized in the served model's tokens: sections differ in token density, so with
`/tokenize` (vLLM, gr8) every final prompt is counted and its length corrected to within 1 % of the target;
without it one ratio from two 1-token requests is used (up to ~10-20 % off) and the table shows the actual prompt
tokens from usage. Sizes above the server's `max_model_len` (when reported) or `--max-context` are skipped and listed. The
warm-up runs one cold prefill of the largest size first. Output: per size, prompt tokens, TTFT cold and cached,
cold prefill tok/s, cached tokens, decode tok/s with a CI over the decode requests, step time and tokens/step.

## Multi-turn agent mode (`run --mode agent`)

```
python3 -m speedbench run --arm A=URL --arm B=URL,extra='{...}' --mode agent --metrics --out ag- [--isolate field]
python3 -m speedbench compare ag-A.json ag-B.json
```

- **Workload:** 8 original conversations (`prompts/speedbench-agent-v1.json`): debugging a failing test suite,
  incident response, data analysis, code review and research/writing with tool calls (`read_file`, `write_file`,
  `run_shell`, `search`), and support, requirements and meeting follow-ups as plain text; 4-8 requests each, 48 in
  total (agent-quick: 4 conversations x 4 requests). Every request repeats the previous request's prompt and appends
  one turn, so the prefix cache can serve everything but the new suffix. A ~3.5K-token system prompt (agent rules
  and a workspace overview) is shared; prompts grow from ~4-7K to ~12K tokens, ~56K new prompt tokens per arm.
- **Frozen history:** the assistant turns in the history are recorded replies, not the model's own output.
  Request t is therefore the same prompt in every arm and run (pairable, its TTFT free of trajectory luck), and only
  each request's own generation is measured (`--max-tokens`, default 160). If the model's replies were fed back,
  each arm would build a different history and later turns could not be paired. Consequence: the frozen reply is
  always part of the new suffix, while in a live agent loop an engine that caches generated tokens may already
  hold its own reply.
- **Cache isolation (`--isolate`):** both arms usually run on one server, so each conversation gets a salt per run,
  arm, repeat and seed. `text` (default) puts a line `Session: <salt>` at the start of the system message: works on
  every engine, but the arms' texts differ, so outputs differ even for identical configurations (no OUTPUT IDENTITY,
  and decode tok/s carries trajectory luck). `field` sends the salt as the `cache_salt` request field (vLLM) and
  keeps the text identical; the run warns when turn 1 is nevertheless served from the cache (the engine ignored the
  field). `none`: only for arms on separate, freshly started servers.
- **Reported per turn** (`report`, run summary): TTFT, prompt tokens, cached tokens
  (`usage.prompt_tokens_details.cached_tokens`, or the `/metrics` delta with `--metrics`), new tokens, cache hit vs
  an ideal cache (the previous request's prompt; engines cache in blocks, so slightly below 1), new-token prefill
  rate, decode tok/s, end-to-end turn latency. No priming request (it would cache the new suffix before it is timed)
  and no sentinels.
- **`compare`:** pairs are turns, but the CI is **clustered by conversation**: the turns of one conversation share
  content, cache state and possibly a conversation-specific effect of the lever, so they are not independent. The
  variance is the between-conversation variance of the per-conversation mean differences / 8 (agent-quick: 4) plus
  the session term; the degrees of freedom are Welch-Satterthwaite over both parts. The conversation part has
  n - 1 = 7 (3) df, but the session term enters with the df of its source: the interleaved residual prior (0.1 %) is a
  fixed assumption with infinite df, so the combined df is above 7 when the session term is not negligible (e.g.
  7.3) and infinite when the conversations agree exactly. The interval therefore leans on the prior when the
  conversation-to-conversation spread is small; with several interleaved sessions the session term is estimated
  from the data (finite df). An effect that differs between conversations is not declared. AGENT block: TTFT of turn 1 (cold), TTFT of turns 2+ (cached) and turn
  latency with CIs and verdicts (lower is better), a per-turn-index table, cache statistics and isolation status
  per arm. Latency is valid for every successful request whatever its output length (short tool calls count);
  decode tok/s is timed for outputs of at least 8 tokens and, when too few conversations have one, the decode
  verdict is "NO DECODE VERDICT" while the latency verdicts stand. Every agent-quick verdict is SCREEN ONLY.
- **Cache isolation status** per arm, checked **per conversation**: *verified* only if the first turn of every
  conversation missed the cache (at most max(64 tokens, 2 % of the prompt) served cached); *FAILED* lists the
  conversations whose first turn was served from the cache (an average would hide one contaminated conversation);
  **UNVERIFIED** lists the conversations without evidence: the first turn failed, or the engine reports no cached
  prompt tokens. Unless isolation is verified in both arms, the agent latency verdicts carry "CACHE ISOLATION
  UNVERIFIED" (or "FAILED"). Cached-token counts are off by default in some
  engines: start vLLM or gr8 with `--enable-prompt-tokens-details`, SGLang with `--enable-cache-report`, or run with
  `--metrics` where `/metrics` exports cached prompt tokens. Unverified isolation with `--isolate field` means an
  engine that ignores `cache_salt` could serve one arm from the other's cache unnoticed.

## Energy (`--power-cmd`)

```
python3 -m speedbench run ... --power-cmd "nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits -lms 200"
python3 -m speedbench run ... --power-cmd "ssh node1 nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits -lms 200" \
                              --power-cmd "ssh node2 nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits -lms 200"
python3 -m speedbench run ... --power-cmd "my-wall-meter --watts" --power-kind wall
```

- Off unless `--power-cmd` is given; then each command runs for the whole run and must print one reading in watts per
  line (numbers on a line are summed; `N/A` lines are skipped). Several commands are summed: one per GPU or node of a
  multi-node server. Local commands run under a pseudo-terminal so line-buffered output arrives promptly; a source
  whose lines arrive in bursts is flagged (use `stdbuf -oL` or `ssh -tt`).
- Readings are timestamped on arrival with the client's clock, interpolated linearly and integrated over each
  request's decode window (first to last text event) and whole request. Reported per category: **J per decode
  token and tok/J**, J per output token of whole requests (prefill included), mean power while decoding, idle power
  (read for `--power-idle` s after the warm-up) and J/token above idle; per run, the energy of the whole measured
  phase per output token. `load`: per concurrency level, block and steady-window J/token. `compare`: a paired CI of
  decode J/token (lower is better).
- **GPU-reported power is not system power:** nvidia-smi `power.draw` covers the GPU board (on unified-memory systems
  such as GB10, what the driver attributes to the GPU), not the CPU, memory, network, fans or power-supply losses.
  Only a wall meter measures the whole system; declare it with `--power-kind wall`. Every report repeats this.
- Alignment is good to about one sampling interval plus transport delay; single short requests are noisy, aggregates
  are not. A window that falls into a gap between readings longer than max(1 s, 5 x the median interval) (a stalled
  source) gets no energy. Per-request energy needs non-overlapping requests (c1).
- Polling the GPU driver can itself cost speed on some systems; compare speed only between runs that sampled power
  the same way (`compare` notes runs that differ). In an interleaved A/B both arms share the sampler.

## Quality guard (`--quality-guard`)

A guard against gross breakage (a wrong chat template, a broken kernel, garbled sampling), **not a quality
benchmark**: 30 easy items cannot measure small quality differences.

- **Exact-answer items** (`prompts/speedbench-guard-v1.json`): 30 short arithmetic, factual, format, code-evaluation
  and translation questions with deterministic graders (last number, whole-word text match, JSON equality), sent at
  T=0 to every arm after the timed workload (`--guard-max-tokens 512`, thinking included; items without an answer
  within the limit are counted separately). `compare` pairs them per item, per session (session i of A with
session i of B, so a later broken session is not hidden by an earlier good one) and flags a drop in B (exact one-sided
  McNemar p < 0.10, or 3+ net losses): "FLAG: B LOST ITEMS".
- **Divergence summary** (T=0, outputs that differ between the arms): the approximate token position where they
  first differ (character position mapped through the chunk trace): min, quartiles and how many diverge before
  tokens 16 and 64. A floating-point-order change usually lets two outputs agree for a while and then flip at a
  near-tie; most outputs differing from the first tokens points to a prompt, template or sampling change. Needs the
  text: `--quality-guard` implies `--keep-text` (any run with `--keep-text` gets the summary).
- OUTPUT IDENTITY (always on) remains the strongest check: all outputs identical = nothing changed the text.

## T>0 with seeds

`--temperature 0.7 --seeds 1,2,3` sends every prompt once per seed (OpenAI `seed`). At T>0 the text depends on the
seed **and** on everything that changes the random draws (drafting and verification, batching, kernels), so every
lever is trajectory-changing and the per-prompt noise is at least that of a T=0 trajectory-changing lever: more
prompts or seeds are needed for the same MDE.

- Pairs are per (prompt, seed): the same seed in both arms (correlated where the engine reproduces seeds), each
  prompt's seeds averaged, the CI over prompts, so the sampling variance is inside it. A short output is a sample,
  not a failure: at T>0 every output with at least 8 decode tokens is timed (T=0 full/quick keep 32).
- **Exclusions can fake an effect** when they depend on the outcome (an arm's slow seeds are also its short or
  failing ones), and the values of dropped outputs cannot be bounded from the data. The rule is therefore simple
  and applies to every mode and every verdict (tok/s, agent TTFT and latency): a pair that failed or is too short
  in either arm is dropped from both and counted, and **any drop makes the verdict NOT PUBLISHABLE** (descriptive
  only, numbers and drop count printed). On top of that the verdict is **WITHHELD** when (a) more than
  max(2, 5 %) of the pairs are dropped, (b) at least 5 single-arm drops fall mostly on one arm (two-sided exact
  binomial p < 0.10), or (c) the printed sensitivity diagnostic (every dropped value imputed at the least
  favourable end of what that arm produced on the prompt's other seeds, or at the least favourable observed paired
  difference, whichever is worse) gives a CI that includes zero or lies on the other side of the verdict.
  Self-test "exclusion trap" (40 prompts, one slow and short seed per prompt, complete-data effect zero, including
  40 x 25 seeds with 4 % dropped in both arms): a publishable B FASTER comes out in 0 % of the simulations.
- `compare` prints a T>0 block: the per-(prompt, seed) SD, the share of the per-prompt variance that is seed
  sampling, and the **MDE with the current, 2x and 4x the seeds**, next to the usual prompts-for-a-1 %-MDE line.

## Results registry

Every command takes `--registry results/registry.jsonl`; `python3 -m speedbench registry add FILE...` adds
existing results. Each line holds the metadata (date, model, quantization, engine, version/commit, configuration
summary, hardware, thinking mode and effort, mode, host compaction, fresh boot, clock policy, notes), the result
file and its SHA-256, and the headline numbers. `python3 -m speedbench registry table --readme README.md`
rewrites the tables between the markers below.

### Results

<!-- speedbench:results:begin -->
_Generated by `python3 -m speedbench registry table` from `results/registry.jsonl`; single runs, descriptive. Decode tok/s = geometric mean over prompts of per-request decode tok/s (c1). An A/B claim needs `compare` on interleaved runs, not two rows of this table._

#### Decode, c1 (full / quick)

| date | label | model | quant | engine | hardware | thinking / effort | mode | tok/s | prose | code | struct. | agentic | multi | longctx | TTFT ms | step ms | tok/step | tok/J | host / clocks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-19 14:39 | gr8 recipe v0.8 | GLM-5.3 | EXLR8-k2k3k4 | gr8 recipe v0.8 6448a16 | 4xGB10-TP4 | low / low | full 256 | 48.9 | 42.8 | 52.9 | 56.4 | 50.9 | 44.2 | 48.6 | 171.8 | 56.1 | 2.66 | – | compacted yes, fresh yes, cap2200 |
| 2026-10-19 14:48 | gr8 recipe v0.9 | GLM-5.3 | EXLR8-k2k3k4 | gr8 recipe v0.9 c53b5b8 | 4xGB10-TP4 | low / low | full 256 | 49.4 | 42.9 | 53.4 | 58.2 | 50.3 | 45.4 | 48.9 | 174.5 | 55.3 | 2.66 | – | compacted yes, fresh yes, cap2200 |
| 2026-10-19 17:12 | GLM-5.3-Flash technigmaai v0.6.0 | GLM-5.3-Flash | NVFP4 (nvidia) | vllm technigmaai v0.6.0 c748079 base | 2xGB10-TP2 | low / low | full 256 | 49.9 | 41.0 | 57.2 | 62.1 | 51.6 | 38.3 | 54.2 | 244.1 | 59.0 | 2.96 | – | compacted no, fresh yes, cap2200 |
| 2026-10-19 17:36 | GLM-5.3-Flash technigmaai v0.1.0 | GLM-5.3-Flash | NVFP4 (nvidia) | vllm technigmaai v0.1.0 c748079 base | 2xGB10-TP2 | low / low | full 256 | 50.2 | 41.1 | 58.1 | 61.9 | 52.1 | 38.2 | 54.8 | 255.4 | 58.1 | 3.02 | – | compacted no, fresh yes, cap2200 |
<!-- speedbench:results:end -->

## Comparison with other benchmarks

Sensitivities below are **estimates** from `selftest` (simulated with the luck measured on our fleet, GLM-5.3 on
gr8), not measurements of those tools.

| benchmark | design | what it controls | false positive at no effect | honest MDE |
|---|---|---|---|---|
| **[RigMark](https://github.com/alexellis/rigmark)** decode workloads | one prompt per category, sent 5x at T=0; repeats treated as samples | timing jitter | **~80 %** (repeats share one trajectory: luck uncontrolled; selftest 78-81 %) | ~7 % |
| **[sparkDash](https://github.com/MiaAI-Lab/sparkDash)-style** dashboards | one prompt per type, one number each; for C>1 the same prompt replicated in every stream | nothing statistical | **~75 %** if a 1 % difference is read as real (one prompt, ~512 tokens assumed, trajectory luck ~3.7 %) | ~10 % (2.8 x luck SD) |
| **decset-style 24-prompt set** | 24 prompts x 384 tokens, mean over prompts | trajectory luck (24 samples) | 5 % with a paired CI, interleaved | ~2.6 % interleaved; ~4.7 % across sessions (one each) |
| **speedbench full** | 73 prompts x 256, paired, interleaved, session term | luck, session drift, failures | 4-6 % (nominal 5 %) | ~1.7 % (tok/s), ~0.4 % (step time, step-cost levers) |

Notes:

- RigMark's within-category repeats look 0.5 % precise but are one trajectory; a lever that changes the text
  draws a new one. Its per-category numbers are not additive across levers for that reason.
- sparkDash-style C>1 numbers replicate one prompt in every stream: at T=0 all streams follow the same
  trajectory, so the replicas average no luck; they measure batching on that one trajectory. Such dashboards
  send `enable_thinking: false`. Full GLM-5.3's chat template ignores that switch (it always opens `<think>` and
  defaults to effort max), so their GLM-5.3 numbers are **thinking-on at max effort**; models with a real
  thinking switch (Qwen, GLM-5.3-Flash) honour it. speedbench sends `reasoning_effort` (default low) and records
  it; record the served thinking mode with `--thinking`.
- Cross-harness numbers are not comparable (prompts, lengths, thinking mode, token boundaries and timing points
  differ); compare like with like.

## Reading the compare report

Example (`run --arm` full, two gr8-flavoured mock servers; the mock's trajectory luck is about twice the real one;
abridged):

```
DECODE tok/s, B vs A (paired over 73 prompts; 0 excluded)
  overall    +1.41 %  [-0.92, +3.79]  p=0.232  df=58   -> NO DETECTABLE DIFFERENCE (MDE80 3.4 %)  [interleaved in one session: boot shocks cancel]
...
DECOMPOSITION (descriptive; steps from metrics)
  step time (client, approx.) B vs A  -2.62 %  [-2.88, -2.36]   per-prompt SD 0.78 %   MDE80 0.38 %
  step time (server)   B vs A  -2.62 %  [-2.88, -2.36]   per-prompt SD 0.78 %   MDE80 0.38 %
  tokens per step      B vs A  -1.25 %  [-3.47, +1.03]   per-prompt SD 9.70 %   MDE80 3.30 %
  draft acceptance     B vs A  -2.08 %  [-5.89, +1.89]   per-prompt SD 16.87 %   MDE80 5.81 %
  drafts per step      B vs A  +0.00 %  [-0.20, +0.20]   per-prompt SD 0.00 %   MDE80 0.28 %
  per category, B vs A [95 % CI] (descriptive): ...

STEP-TIME RULE (pre-registered with run --preregister step; declared; margin 2.0 % from pre-registered; on server step time)
  [ok] workload valid (no WITHHELD verdict): NO DETECTABLE DIFFERENCE (MDE80 3.4 %)
  [ok] arms interleaved in one session: shared and interleaved
  [ok] steps from engine counters: sources: metrics
  [ok] server-side step time in every request: engine decode time per step
  [FAIL] tokens/step equivalent within +-2.0 %: 90 % CI [-3.11, +0.65] %
  [ok] drafts/step equivalent within +-2.0 %: 90 % CI [-0.16, +0.16] %
  [ok] tok/s verdict not contradicted: tok/s CI [-0.92, +3.79] %
  step-time verdict: B FASTER (step time lower); implied tok/s effect at equal acceptance +2.69 % [+2.42, +2.96]
  tok/s effect allowing for the measured acceptance uncertainty: [-0.77, +3.64] %
  -> DECISION: the tok/s verdict above (a check failed)
```

(The mock lever made steps 3 % cheaper and moved every prompt to a new trajectory. The step time sees it
clearly; with the mock's exaggerated luck the tokens/step change cannot be bounded within 2 %, so tok/s decides.)

- **overall** is the decision unless the step-time rule applies: geometric-mean speed change, 95 % CI, p-value,
  verdict. A `CROSS-SESSION` prefix means the arms ran in different sessions (see "Sessions").
- **per-prompt SD** tells you which regime you are in: ~0.5 % = bit-exact lever; ~4-6 % = trajectory luck.
- **boot term**: shared session (residual prior, or estimated from >= 2 interleaved sessions), estimated
  (>= 2 sessions per arm, pooled with the prior), or assumed ("prior-conditional").
- **OUTPUT IDENTITY** counts prompts whose output is byte-identical in every A and B request. All identical:
  no trajectory luck, and the tok/s verdict is also the step-time verdict.
- **DECOMPOSITION**: B vs A for step time (client and server), tokens/step, acceptance and drafts/step,
  overall and per category; log tok/s = log tokens/step - log step time per request.
- **TTFT** rows: the measured request runs with the prompt cached by the priming request (warm); the priming
  request's TTFT is the prefill of a prompt not yet in the cache on a first pass.
- **HORIZON**, **sentinel** lines: as in v0.1 (luck vs output length; drift and determinism of re-run jobs).

## Validation

- `python3 -m unittest discover -s tests -t .` (90 tests, ~90 s): statistics (t quantiles, pairing, clustering,
  frozen weights, session terms, CI coverage, boots not counted as luck samples, ANOVA of interleaved sessions),
  prompt set reproducibility, SSE parsing, protected request fields, compare input checks, Prometheus parsing and
  engine detection, per-request SGLang/gr8 fields, the decomposition identity, and end-to-end runs against mock
  servers: a faster engine with different trajectories is detected, same-config CIs cover zero, gr8-style
  counters at request end give engine steps, the step-time rule, cross-session banners, SGLang per-request stats,
  concurrency scaling and batch occupancy, the agentic mix and its split at state changes, no window throughput
  without per-chunk usage, long-context cold vs cached (the salt defeats the cache; per-document sizing within
  2 %; sizes above max_model_len are skipped), report CIs over prompts (repeats are not samples), the step-rule
  gates (an acceptance loss cannot flip a slower tok/s; not interleaved and withheld runs never decide), the
  registry/README table writer, and (v0.3) the agent set (every request extends the previous one; 48/16 requests),
  cache isolation on one shared server (turn 1 cold in both arms, later turns cached; `cache_salt` keeps the text
  identical; an ignored `cache_salt` is detected), power integration and summing of two fake power commands
  (J/token = W / tok/s; no energy fields when off), the guard graders on every shipped item, a guard drop flagged
  end to end, the divergence position, T>0 pairing per (prompt, seed) with dropped pairs, the seed MDE projection,
  and self-test coverage with seeds; and one regression test per finding of the v0.3 review (conversation
  clustering, latency independent of output length, the exclusion trap, sampling gaps in the power trace, unverified
  cache isolation, guard per session, tool names in the divergence, SCREEN ONLY on agent-quick, load energy scope)
  and its verification round (drops below the cap are not publishable, the sensitivity check keeps the direction,
  one decode-validity predicate for the boundary check and the analysis, per-conversation cache isolation) and the
  final gate (step-time decision needs complete coverage; failed first turns are no isolation evidence).
- `python3 -m speedbench selftest [--sims 300]`: the coverage/power table above.
- `python3 -m speedbench mock --port 8399 --step-ms 20 --numerics 1 [--flavor vllm|gr8|sglang] [--proposed]`: a
  mock OpenAI server with one engine thread (batched decode steps, chunked prefill, a prefix cache honouring
  `cache_salt`, per-trajectory acceptance luck, context-dependent step cost) and the `/metrics` and per-request
  fields of each engine; it answers the guard items (`--break-answers F` makes a fraction wrong) and can share the
  T=0 text between numerics up to a per-prompt position (`--diverge 0.5`).

Not done yet: runs against a real engine (the fleet was busy). On the first real run check: 0 failed requests;
"token boundary exact"; `steps_source` metrics (gr8, vLLM) or response (SGLang); stream events vs step counters;
sentinels identical; an A-vs-A interleaved session covers zero; the long-context token calibration (prompt tokens
within a few per cent of the target); agent turn 1 not cached and turns 2+ cached; the guard passes at effort low
within 512 tokens; power readings not buffered (`buffered_frac`).

## Prompt set

`prompts/speedbench-v1.json` (73 prompts), `prompts/speedbench-agent-v1.json` (8 multi-turn conversations) and
`prompts/speedbench-guard-v1.json` (30 guard items); version and SHA-256 recorded in every result. All prompts are
original; the long-context documents (meeting minutes, logs, code, sales CSV, support chat, requirements, a
chronicle, configuration, a diff) are generated from fixed seeds by `tools/build_prompts.py`, which also checks
that it reproduces the shipped files (`--check`); `longctx` and the agentic mix use the same generators at
larger sizes. Never edit the JSON by hand: results from different prompt-set versions are not comparable.

## Result files

- `speedbench-result/1` (`run`): `server`, `settings`, `prompt_set`, `meta` (operator metadata), `summary` (per
  category: median, geometric mean and aggregate tok/s, TTFT, reasoning share, step time, tokens/step,
  acceptance, sentinels, `energy`; `agent` per turn in the agent modes) and `requests` (per request: tok/s, TTFT,
  decode time, token counts, finish reason, step counts and decomposition, engine counter deltas `spec` /
  per-request `spec_request`, output hash, priming TTFT, wall time, chunk trace `[ms since send, chars, cumulative
  tokens or -1, channel]`; agent: `conv`, `turn`; energy: `energy_j`, `decode_energy_j`, `j_per_tok`). Optional
  `energy` (sources, raw samples, idle power, note on what is measured) and `guard` (per item: pass, extracted
  answer, tail of the reply).
- `speedbench-load/1` (`load`): `levels` (per c; `energy` with `--power-cmd`), `mix`, `requests`, `energy`.
- `speedbench-longctx/1` (`longctx`): `table` (per size), `requests`.

## Limitations

- Client-side timing: network and SSE framing are included; run the client next to the server.
- Fixed horizon: with thinking at effort low, the first 256 tokens are mostly reasoning. It measures
  thinking-model decode speed, not time-to-answer or task completion.
- The session prior (1 %) rests on one observed ~2 % shift; with one session per arm every verdict leans on it.
  Interleave, or run several sessions per arm so it is estimated.
- The step-time SD used in the sensitivity table (0.6 %) is assumed; `compare` prints the measured one.
- c>1 and the agentic mix are descriptive; neighbours' prefill and decode phases interfere by design.
- Tuning a lever on this prompt set and judging it on the same set overfits; confirm finalists with
  `--variants` (new trajectories) or a different seed set.
- Agent mode replays a frozen history: it measures prefill of the new suffix and decode per turn, not a live agent
  loop (no tool execution, the model's own replies are not cached as history). With 8 conversations its CIs rest on
  few clusters and, when the conversations agree closely, on the session prior.
- Energy is only as good as its source: GPU-reported power excludes the rest of the system, sampling intervals
  limit alignment, and polling can cost speed.
- The quality guard detects gross breakage only.

## License

Apache-2.0. Prompts and generated documents are original works released under the same license.
