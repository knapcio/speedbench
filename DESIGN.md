# Design and review record

This file summarises how the benchmark was designed: the starting problem, the calibration data, the rounds of
independent critique, what changed, and where the reviewers disagreed (with each other or with the design)
and how each disagreement was settled.

Reviewers:
- **Reviewer A**: worked with full access to the repository; ran the tests and its own simulations and fixtures.
- **Reviewer B**: an independent second reviewer; re-derived the calibration and ran its own simulations.

Process: design v0 -> round 1 critiques (A, B) -> v1, implemented -> round 2 critiques of the code (A, B) -> v1.1
-> v0.2 (new modes, session handling; round 3 critique and a verification round by A, see below) -> v0.3 (agent
mode, energy, quality guard, T>0 pairing; not yet reviewed, see the end of this file).

## Problem and calibration

[RigMark](https://github.com/alexellis/rigmark)'s decode workloads send one prompt five times at T=0. Its repeats share one trajectory, so they average
timing jitter but not trajectory luck: any change of the T=0 text moves draft acceptance and tok/s by luck.

Calibration from our test cluster (decset.py runs, 2026-10-16..19, GLM-5.3 full on gr8 TP4; 36 configurations,
99 runs of 24 prompts at 384 tokens; `calibration/decset-runs-2026-10.json`). Per-prompt SD of the paired log
speed ratio:

| comparison | SD |
|---|---|
| trajectory-changing levers (dselA->dselB, v07->v09r, v09r->v09d, s9->s10) | 3.5-5.1 %, pooled 4.27 %, kurtosis ~6, extremes -18.8/+8.6 % |
| bit-exact levers (kb0->kb8, rA->rB, v07->l2g210) | 0.22-0.52 % |
| boot shift of a whole run (two-way ANOVA per configuration) | 0.05-0.65 %, median ~0.18 %, RMS ~0.27 % |

Reviewer B found no stable "lucky prompt" across levers (cross-lever correlation of centred per-prompt luck between
-0.5 and +0.58), i.e. luck belongs to the trajectory, not to the prompt.

## Round 1 (design v0) - main findings and decisions

| finding | by | decision in v1 |
|---|---|---|
| +1 % is not detectable in 10 min for trajectory-changing levers by brute force (N=73 x 256: power ~0.3-0.45) | both | accepted; the tool prints the achieved MDE and what 1 % would cost instead of promising it |
| first chunk can carry several tokens; numerator and interval must use the same boundary | both (A: P0) | exact boundaries from per-chunk cumulative usage; the maintainer's formula (n-1) only as a declared fallback |
| boot variance: a fixed floor or a 1-df estimate gives dishonest CIs; sign-flip test ignores boot shocks | A (P0) | two-way ANOVA per arm pooled with a prior; t with Satterthwaite df; permutation p removed |
| category verdicts = multiple-comparison bait | B | one overall verdict; categories descriptive |
| determinism must be detected, not assumed | B | OUTPUT IDENTITY block, sentinels |
| decompose tok/s into step time x tokens/step | B | implemented, descriptive (see disagreement 1) |
| matched-prefix estimator selects trajectories | A | cut |
| cache state changes gr8's T=0 text | B | priming request on by default |
| engine adapters: `reasoning` vs `reasoning_content`, tool-call deltas, usage-only chunks; never estimate tokens from characters | A | implemented; missing usage = failed request |
| 1/sqrt(L) scaling unverified | both | HORIZON block measures SD at 64/128/192 tokens on real runs; the self-test applies sqrt(L) only where it is conservative |

## Round 2 (implementation) - main findings and fixes

| finding | by | fix in v1.1 |
|---|---|---|
| tokens after the last text event (EOS, hidden tokens) counted against a shorter interval: +0.4 % at 256 tokens, worse when finish reasons differ | both | decode tokens = cumulative count at last text event - at first text event |
| truncated stream (no finish_reason) accepted as a measurement; mid-stream timeout escaped and could lose the run | A | completion contract in the client; every request exception becomes a failed request; expected job counts checked |
| the same file given twice counted as two boots (turned +0.4 % NO DIFFERENCE into B FASTER) | A | duplicate path/content/run_id rejected; `--block` groups runs of one boot |
| a failed variant silently changed a prompt's mean (0 % -> +10.5 %, B FASTER); weights followed survivors | A | a prompt is used only with all its variants/seeds; frozen category weights; verdict withheld above max(2, 5 %) excluded prompts |
| `--extra-body` could override temperature/max_tokens/effort (even the 1-token prime) unseen | A | protected fields refused; effective request recorded and compared |
| default labels collided in label-keyed summaries | A | run ids; labels default to the file name |
| per-chunk usage on in one arm and off in the other fabricates 1-2 % | B (P0) | `compare` refuses mixed token boundaries |
| boot prior applied to interleaved in-session A/B, where the shock cancels: CI 3x too wide for bit-exact levers | B (P0) | session id; interleaved arms get a residual term (0.1 %) instead |
| prior-pooled boot term is prior-dominated; true boot SD 0.5 % -> coverage 0.68-0.86 | both | labelled "prior-conditional", CI under 0.5 % shown, warning when the estimate exceeds 2x the prior, README: calibrate the prior, runs = boots |
| stream events are not engine steps on gr8 (empty events are dropped) and with buffering tool parsers | A, B | engine counters (`--metrics`) preferred; agreement printed; stream-only decomposition flagged |
| self-test drift was identical in both arms (cancels) | A | drift now applied to one arm (shows bias 0.87 coverage, which is what sentinels are for) |
| variants: tag in a system message changes the template path; "halves the CI" overpromised | B, A | tag on the last user message; correlated-variant scenario simulated (CI 1.35 -> 1.06 %, not 0.68 %) |
| MDE used normal quantiles; singleton categories had zero variance | both | t quantiles; singleton categories borrow the pooled variance |
| quick mode verdicts would be misused | B | labelled "SCREEN ONLY" |

## Disagreements and how they were settled

1. **Step time as a primary verdict.** Reviewer B (round 1) wanted step time as the verdict for kernel levers
   because it is nearly luck-free; reviewer A called acceptance a mediator and wanted decomposition descriptive.
   The maintainer's headline is end-to-end tok/s. Settled in round 2 with B partly conceding: tok/s stays the
   only verdict; DECOMPOSITION carries a predeclared rule (step time is decision-grade only if the
   tokens-per-step CI includes 0 and |estimate| < 0.5 %), and for bit-exact levers the report states that the
   tok/s verdict is the step-time verdict.
2. **Primary estimator.** Reviewer B (round 1) proposed a 3-MAD winsorized mean (about 10 % smaller SE); A
   argued that tail regressions are real effects of a configuration and robustness chosen after seeing the data
   changes answers (s9->s10: mean +0.73 %, median +1.72 %). Mean log ratio kept as the predeclared estimator;
   winsorized, median and Hodges-Lehmann are descriptive. B withdrew the proposal in round 2.
3. **The k1 = 1 fallback.** Reviewer A wanted the decode verdict withheld unless token/time boundaries are
   validated. The maintainer defined the headline as (completion_tokens - 1)/(last - first token), so the fallback stays,
   but it is labelled ("token boundary assumed"), never mixed with exact runs, and exact counting is used
   whenever the server provides per-chunk usage (gr8 and vLLM do).
4. **Boot prior.** Reviewer A: pooling with a prior is regularisation, not a coverage guarantee; require 4-6
   independent paired boot blocks for unconditional claims. Reviewer B: auto-raise the prior to 0.5 % when an arm is
   nondeterministic. Kept the 0.25 % prior (our data: median 0.18 %, RMS 0.27 %) as a declared assumption, with
   the 0.5 % CI printed next to it, a warning on large estimates, and the interleaved mode where the prior does
   not matter. B's auto-raise was not adopted: in our ANOVA, nondeterminism shows up in the residual (which
   is already in the per-prompt variance), not in the boot term.
5. **Prompt-set size.** Reviewer B (round 1) suggested ~56 prompts (fewer tool, multilingual and long-context
   prompts, which add prefill and parser risk) plus seeds. The maintainer requires those categories; kept 73
   (an early Polish category was dropped: the set is meant to be universal), with
   `--skip-tools` and a tool probe that stops a run before measuring if the server cannot handle tool history.
6. **Mock server.** Reviewer B (round 1) called the spec-decode mock overbuilt (resampling the real luck is enough
   for the statistics). The maintainer asked for a mock that simulates trajectory luck; both exist: the self-test
   resamples real luck, the mock exercises the client end to end. Reviewer A (round 2): mock step counting does not
   validate a real engine - agreed, the README lists what to check on the first real run.
7. **Priming.** Reviewer B wanted it on (same cache state, gr8's T=0 text depends on it); A noted that it does not
   guarantee equal state and that the priming TTFT is only "cold" on a first pass. On by default, cached-token
   counts recorded, the TTFT rows described accordingly.
8. **Concurrency.** Reviewer A suggested deferring c2/c4 until c1 is trusted; the maintainer asked for it. Kept as a
   descriptive mode (no priming, no sentinels, statistics assume c1).

## v0.2: decomposition, sessions, new modes (round 3)

Requirements: decompose speed into causes from engine counters (gr8, vLLM, SGLang) with a pre-registered
rule for when step time decides; concurrency and an agentic mix; long contexts (TTFT 8K..128K cold and cached,
decode at 32-128K); a results registry and a comparison with other benchmarks. Fleet findings that changed the
statistics:

| finding (test cluster, 2026-10-19) | consequence |
|---|---|
| same bit-exact config ~2 % apart across two sessions, ~0.1 % within one | A/B arms interleaved in one session; cross-session comparisons labelled and bannered; boot (session) prior 0.25 -> 1.0 % per arm, CI under 2 % shown; >= 2 interleaved sessions: session x arm residual estimated by ANOVA on per-session differences |
| T=0 text is identical on every boot | luck is fixed per configuration and variant: extra boots average timing and session noise only (prompt term divided by prompts, not by boots; regression test) |
| fragmented host memory made one rank 20-80 % slower | host preparation section; `--host-compacted`, `--fresh-boot` recorded |
| first prefill after a boot is slow | every mode's warm-up includes a long cold prefill (8K in run, the long document in the mix, the largest size in longctx) |
| clock policy matters | `--clock-policy` recorded; `compare` notes metadata that differs |

Decisions:

1. **Decomposition per request, exact.** tokens/step = (completion - 1) / engine steps and step time = decode
   interval / steps inside it, so log tok/s = log tokens/step - log step time holds per request. Engine steps come
   from per-request response fields (SGLang `sglext`, proposed gr8 `usage.step_stats`) or the `/metrics` delta at
   c1; a delta during which a different number of requests finished is discarded (`spec_note`). gr8 adds its
   counters at request end, which is before the client sees `[DONE]`, so c1 deltas are exact there.
2. **Step-time rule, pre-registered.** Declared in the run (`--preregister step`), applied in `compare` only if the
   steps are engine-counted and tokens/step and drafts/step are equivalent within +-2 % (TOST, 90 % CIs). A post-hoc
   request is shown but never decides. The tok/s verdict remains the default decision. This settles the round-1
   disagreement 1 (step time as a verdict) for the class of levers where reviewer B's argument holds.
3. **gr8 gap documented, not patched.** gr8 exposes per-request counters only via `/metrics` at request end and
   no engine-wide step counter; the smallest upstream addition is written up in
   `docs/gr8-step-counters-proposal.md` (P1 shim + front end, P2 three engine counters); speedbench reads both.
4. **Concurrency and the mix are descriptive** (shared engine, dependent requests): aggregate tok/s over a steady
   window with a CI over time bins, per-stream tok/s, TTFT; the mix splits the long session's decode speed by the
   short requests' state.
5. **Long context uses generated documents** with a salt on the first line (cold) and the same document with
   another question (cached), sized in the served model's tokens by `/tokenize` or two 1-token requests.
6. **Registry** in `results/registry.jsonl` (tracked; result files are not), README tables between markers.
7. **Mock server** now has one engine thread (batched steps, chunked prefill, prefix cache, per-flavour metrics),
   so concurrency, mix and long-context code paths are tested end to end.

### Round 3 review (reviewer A) and fixes

| finding | priority | fix |
|---|---|---|
| step rule could turn a significant "B SLOWER" into "B FASTER" (acceptance loss inside the 2 % margin); margin adjustable after the fact | P1 | the rule may only confirm a significant tok/s result, never overturn or neutralise it; the margin is pre-registered in the run (`run --tps-margin`) |
| step decision bypassed WITHHELD and quick-mode labels | P1 | valid workload and interleaving are checks of the rule; quick-mode step decisions are SCREEN ONLY |
| NOT INTERLEAVED was printed but the 0.1 % residual still used | P1 | the residual path needs shared AND interleaved; otherwise session prior, prior-conditional verdict |
| two-session ANOVA pooled with 4 prior df: ~26 % false positives when the residual is 3x the prior | P1 | kept as a declared prior-conditional estimate; data-only CI printed next to it; warning when the estimate exceeds 2x the prior; README states the stress result |
| client step time infers steps inside the interval from tokens (0.85 % bias in a repro) | P1 | the rule decides on the engine's own decode time per step (`step_ms_server`, matching boundaries); the client value is labelled approximate and never decides (so SGLang cannot use the rule) |
| long-context documents 10-20 % off target (section token density differs) | P1 | every final prompt is counted with `/tokenize` and corrected to within 1 %; warm-up uses the calibrated ratio |
| load fabricated token arrival counts without per-chunk usage | P2 | window throughput and the mix split require per-chunk usage; `block_tok_s` otherwise |
| mix intervals classified by one midpoint | P2 | intervals split at every state change; tokens spread by time (assumption stated) |
| report CI counted repeats as samples | P2 | one value per prompt (geometric mean of its requests) |
| vLLM iteration histogram called "engine steps" | P2 | renamed "output iterations", a lower bound |
| proposed gr8 step counters: launch-to-fence time misses drafting; status.s2 is not a publication barrier | P2 | proposal reworked: round time start-to-start (drafting inside), prefill time separate, totals published through each request's `m` line |
| missing usage.prompt_tokens aborted longctx | P2 | None-safe summary with a note |
| full/quick "unchanged" while the warm-up gained an 8K prefill and a metrics probe | P2 | probe only with `--metrics`; README states +~6 s and how to remove it |

Verification round (reviewer A): 9 findings FIXED, 4 conditional or partial, 3 new. Further fixes: the
document sizing searches inside a bracket and always returns the count of the returned text (with a convergence
flag; prompts that would exceed the context are skipped and listed), and the warm-up uses it too; the step rule
needs one recorded margin and a tok/s point estimate in the step verdict's direction; the gr8 proposal closes the
open round interval before each publication.

Disagreement kept: reviewer A asked that the step decision also require the acceptance-adjusted bound (step-time CI
plus the tokens/step 90 % CI) to exclude zero. That bound is about as wide as the tok/s CI for a
trajectory-changing lever, so the rule would add nothing; the maintainer asked for exactly the acceptance-independent
step-time comparison. Kept as a declared assumption (a step-cost lever does not change acceptance in
expectation), bounded by the equivalence check, voided when the measured acceptance change outweighs the step
gain, with the adjusted bound always printed. The two-session ANOVA stays prior-conditional (25.8 % false
positives when the residual is 3x the prior, disclosed, data-only CI printed).

## v0.3: multi-turn agent mode, energy, quality guard, T>0 pairing

Requirements: an agentic workload that exercises the prefix cache turn by turn; energy per token; a guard that a
speed lever did not break the outputs; correct handling of T>0 with seeds. Decisions:

1. **Agent mode replays a frozen history.** 8 original conversations (debugging, incident response, data analysis,
   code review, research and writing with tool calls; support, requirements and meeting follow-ups as text), 4-8
   requests each, 48 in total (`prompts/speedbench-agent-v1.json`, generated from fixed seeds by
   `tools/build_prompts.py`). Request t = system message + task + the first t-1 turns, each turn a recorded
   assistant reply (often a tool call) plus the next tool result or user message. Feeding the model's own replies
   back was rejected: every arm would build a different history, so later prompts (and their lengths) would differ
   and could not be paired; the prefill side of the comparison would then carry the decode side's trajectory luck.
   With the history frozen, request t is the same prompt in every arm, its TTFT is free of trajectory luck, and only
   the generation of each request is measured. Known cost: an engine that caches its own generated tokens would find
   part of the real reply cached in a live agent loop; here the frozen reply is always new suffix. Contexts grow from
   ~4-7K tokens (a ~3.5K-token system prompt with a workspace overview) to ~6-12K; ~56K new prompt tokens per arm.
2. **Cache isolation by salt.** Arms usually share one server, so arm B would find arm A's turn in the cache. Each
   conversation gets a salt per run, arm, repeat and seed: by default a first line in the system message (works
   everywhere, but the arms' texts differ, so OUTPUT IDENTITY and the divergence summary do not apply and decode tok/s
   carries trajectory luck even for a bit-exact lever), or the `cache_salt` field (vLLM; identical text; the run
   checks that turn 1 is not served from the cache and warns otherwise), or none (separate fresh servers).
3. **Pairs per turn, clusters per conversation** (revised after review, below). TTFT of the cached turns (t >= 2)
   and end-to-end turn latency get their own CIs and verdicts (lower is better). No priming request (it would put
   the new suffix in the cache) and no sentinels (a re-sent turn would be cached).
4. **Energy is integrated, not sampled per request.** Power sources are shell commands printing watts per line
   (nvidia-smi locally or over ssh, or any meter), summed over sources, timestamped on arrival with the client's
   clock, linearly interpolated and integrated over each request's decode window and whole request. Readings that
   arrive in bursts (buffered output) are flagged; windows not covered by readings get no energy. Reported: J per
   decode token and tok/J per category, J per output token of whole requests, idle power and the share above idle,
   and in `compare` a paired J/token CI. GPU-reported power excludes the rest of the system; the report says so
   unless the operator declares a wall meter. Per-request energy needs non-overlapping requests (c1); `load` reports
   per-level block and steady-window J/token. Polling the driver can cost speed, so `compare` notes runs that
   sampled power differently.
5. **Quality guard, explicitly not a quality benchmark.** 30 short exact-answer items (arithmetic, facts, formats,
   code evaluation, one translation) with deterministic graders, run at T=0 in each arm after the timed workload;
   per-item pairing, exact one-sided McNemar test, FLAG when B loses items (p < 0.10 or 3+ net losses). For T=0
   outputs that differ between the arms, the first differing token (character position mapped through the chunk
   trace) is summarised: late divergence is typical of numerics changes; divergence within the first tokens in most
   prompts points to a template, prompt or sampling change. Needs the text, so `--quality-guard` keeps it.
6. **T>0: pairs per (prompt, seed).** Keys carry the seed; each prompt's seeds are averaged and the CI is over prompts,
   so it contains the sampling variance. The all-variants rule for T=0 (a failed variant drops the whole prompt,
   because variants differ systematically) was too strict here: one short seed dropped the prompt, and three seeds
   tripled the exclusion rate (a mock run was WITHHELD for 3/30 prompts). At T>0 the seeds are exchangeable draws, so
   a (prompt, seed) pair missing in either arm is dropped from both and counted (the withholding rules were tightened
   after review, below). `compare` prints the per-pair SD, the share of the per-prompt variance that is seed
   sampling, and the MDE projected for 2x and 4x the seeds. Self-test: 3 seeds with independent trajectories per
   seed cover at 0.96 (full) and 0.95 (quick 30), CI +-0.73 % on the full set (+-1.24 % with one seed).

### v0.3 review and fixes

An independent review of v0.3 blocked it. Findings and fixes, each with a regression test that reproduces the
reviewer's scenario (`tests/test_review_v03.py`):

| finding | priority | fix |
|---|---|---|
| agent CIs treated the turns of a conversation as independent; with conversation-specific effects a cached-TTFT slowdown came out significant | P1 | pairs are turns, the CI is clustered by conversation (8 clusters, df <= 7); README claims restated |
| T>0: outcome-dependent exclusions (one slow, short seed per prompt, 10 % of 400 pairs) reported B FASTER +2.0 % although the complete-data effect was zero | P1 | short outputs are samples at T>0 (timed from 8 tokens); withheld when > 5 % pairs dropped, when single-arm drops are imbalanced, or when a per-prompt worst-case imputation makes the CI include zero; self-test "exclusion trap" (tightened in the verification round, below) |
| published history would expose reviewer identities | P1 | handled outside the code: the public repository starts from a fresh history |
| agent latency depended on the 32-token decode minimum: short tool calls suppressed the latency verdicts, and < 3 valid decode prompts aborted compare | P1 | latency is valid for every successful request; decode without enough timed outputs reports NO DECODE VERDICT and the rest of the report stands |
| energy integration accepted a request inside a 100-s sampling gap | P2 | every gap between the readings around and inside a window must be short |
| cache isolation silently unverified when the engine reports no cached tokens | P2 | isolation status verified / FAILED / UNVERIFIED per arm; engine flags documented |
| multi-run compare used only the first session's quality guard | P2 | guard compared per session pair |
| divergence ignored changed tool names | P2 | function names are part of the compared stream |
| agent-quick latency verdicts lacked SCREEN ONLY | P2 | every agent-quick verdict is labelled |
| load energy reports omitted what the power source measures | P2 | scope note and source warnings in the load report |

Verification round: 8 of 10 fixed; it blocked on three P1 and two P2 findings, closed with simple, conservative rules
rather than more imputation:

| finding | priority | fix |
|---|---|---|
| exclusions below the 5 % cap still manufactured significance: 40 prompts x 25 seeds, one slow seed per prompt untimeable in both arms (4 % dropped), complete-data effect zero -> B FASTER +0.93 % [+0.74, +1.13] | P1 | **any outcome-dependent drop makes a verdict NOT PUBLISHABLE** (descriptive only, numbers and drop count printed), in every mode and for every verdict; a publishable verdict requires zero drops. The cap, attribution test and sensitivity bound remain as WITHHELD rules and printed diagnostics; self-test trap includes the 40 x 25 case |
| the sensitivity check accepted a reversal (B FASTER +1.18 % with a sensitivity CI of -8.45 %) | P1 | the sensitivity CI must keep the verdict's direction: including zero or lying on the other side withholds it |
| short outputs admitted by the analysis (>= 8 tokens) bypassed the boundary check (>= 32): 25 tokens in 1 s counted as 20 (exact) vs 24 (fallback) gave B FASTER +20 % | P1 | one predicate (`compare.decode_valid`) decides "timed decode" for the boundary check and the analysis |
| "verified" isolation averaged over conversations: one fully cached first turn among eight read as 12.5 % | P2 | isolation checked per conversation; FAILED lists the conversations |
| the README promised at most 7 df for agent CIs; Satterthwaite gives 7.3, or infinity with the fixed residual prior | P2 | README describes the actual df and its dependence on the session prior |

Final gate: the step-time rule used the old 32-token validity and decided on a metric that silently missed pairs
(40 x 25 seeds, one 20-token output per prompt without engine telemetry: -2.00 % B FASTER, "DECISION: the step-time
verdict"); it now uses the shared predicate and requires every decision metric to cover every pair of the decode
analysis. Failed first turns counted as isolation evidence ("verified ... (0)" with all eight failed); a
conversation without a successful first turn with cached-token counts is now UNVERIFIED, and the agent latency
verdicts carry the qualification.

Strict worst-case imputation is unbounded (a dropped output can be slower than anything observed); that is why
publication requires zero drops instead of a bound.

## Open items

- First run against a real engine (the fleet was busy): checks listed in the README; for v0.2 also the
  `steps_source`, `spec_note` counts, the SGLang streaming placement of `sglext`, and the long-context token
  calibration.
- Measure the per-prompt step-time SD of a trajectory-changing step-cost lever on the fleet (the sensitivity
  table assumes 0.6 %) and the in-session residual over several interleaved sessions (prior 0.1 %).
- Upstream the gr8 counters (docs/gr8-step-counters-proposal.md).
- Measure on real data: the HORIZON curve (does luck decay with output length?), the variant correlation,
  and the boot SD of the current stack (A vs A on separate boots).
- A held-out confirmation prompt set, to avoid tuning levers on the screening set (reviewer A).
- gr8 token-ID streaming could give exact per-token timing; not used (OpenAI-compatible fields only).
- To check first on a real engine: agent turn-1 cached tokens ~0 and
  turns 2+ near the previous prompt (block granularity), whether `cache_salt` is honoured, the agent runtime, the
  guard pass rate at effort low within 512 tokens, power-source buffering over ssh, and the speed cost of polling.
- Energy per request is not attributed at c>1 or in `longctx` (block totals only in `load`).
- The per-trajectory luck at T>0 is assumed equal to the T=0 luck in the self-test; measure it.
- Agent mode has only 8 conversations: more conversations would give its CIs more clusters and less dependence on
  the session prior.
- Requiring zero drops for a publishable verdict will often label real full-mode runs NOT PUBLISHABLE (a prompt that
  ends early in one arm); measure how often on a real engine before deciding whether a declared, pre-registered
  exclusion rule (e.g. by the A arm's output length only) is worth adding.
