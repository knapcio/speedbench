"""speedbench selftest: does the confidence interval hold under trajectory luck?

Simulates many A/B experiments at the level of per-prompt log decode speeds and runs the same analysis code
(stats.paired) on each. Trajectory luck is drawn from the real per-prompt paired log ratios of trajectory-
changing levers measured on our fleet (calibration/decset-runs-2026-10.json: heavy tails, outliers to -18 %),
or from a normal with the same SD. Boot shocks, per-request timing noise, nondeterministic T=0 and in-run
drift are added on top. For each scenario it reports the CI coverage of the true effect, the false-positive
rate at zero effect, and the power at +1 % and +2 %. A RigMark-style design (one prompt sent 5x) is
simulated the same way to show why it cannot separate a lever from luck.

  python3 -m speedbench selftest [--sims 400] [--quick]
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

from . import stats as S

CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration", "decset-runs-2026-10.json")
TRAJ_PAIRS = [("dselA", "dselB"), ("v07", "v09r"), ("v09r", "v09d"), ("s9-best", "s10-oldtab")]


def empirical_luck(path=CAL):
    """centred per-prompt log ratios of trajectory-changing pairs (per pair and category), pooled"""
    with open(path, encoding="utf-8") as f:
        R = json.load(f)
    cat = ["p"] * 12 + ["c"] * 6 + ["s"] * 6
    out = []
    for a, b in TRAJ_PAIRS:
        ma = [statistics.fmean(math.log(r[1][i]) for r in R[a]) for i in range(24)]
        mb = [statistics.fmean(math.log(r[1][i]) for r in R[b]) for i in range(24)]
        d = [y - x for x, y in zip(ma, mb)]
        for c in set(cat):
            v = [d[i] for i in range(24) if cat[i] == c]
            m = statistics.fmean(v)
            k = math.sqrt(len(v) / (len(v) - 1))       # undo the shrinkage of centring within a small group
            out += [(x - m) * k for x in v]
    return out


def simulate(rng, n, effect, luck, runs=1, boot_sd=0.01, req_sd=0.004, nondet=0.0, drift=0.0,
             n_cats=6, boot_prior=0.01, rigmark=False, shared=False, residual_sd=0.0007, shared_prior=0.001,
             variants=1, var_corr=0.3, sparkdash=False):
    """one experiment -> analysis result. luck: callable returning one trajectory-luck draw (log ratio).

    drift: in-run slowdown of arm B only (e.g. B ran on a warming fleet), linear over the run.
    shared: A and B interleaved in one session on one boot per run (same shock; residual_sd remains).
    variants: trajectories per prompt; their luck shares a fraction var_corr of its variance (same answer mode).
    Trajectory luck is drawn once per prompt and arm and reused in every run: at T=0 a configuration produces
    the same text on every boot, so extra boots are not extra luck samples.
    sparkdash: one prompt, one request per arm, read as "different" when the two numbers differ by > 1 %.
    """
    cats, cluster = {}, {}
    for i in range(n):
        for v in range(variants):
            k = "p%02d@v%d" % (i, v)
            cats[k], cluster[k] = "c%d" % (i % n_cats), "p%02d" % i
    keys = sorted(cats)
    base = {k: rng.gauss(math.log(45), 0.15) for k in keys}
    # A arm keeps trajectory 0; B arm's numerics move every trajectory (luck relative to A)
    common = {c: luck() for c in set(cluster.values())}
    luckB = {k: math.sqrt(var_corr) * common[cluster[k]] + math.sqrt(1 - var_corr) * luck() if variants > 1
             else common[cluster[k]] for k in keys}
    shocks = [rng.gauss(0, boot_sd) for _ in range(runs)]

    def run(arm, r):
        shock = shocks[r] + rng.gauss(0, residual_sd) if shared else rng.gauss(0, boot_sd)
        out = {}
        for j, k in enumerate(keys):
            traj = luckB[k] if arm == "B" else 0.0
            if nondet and rng.random() < nondet:       # this request landed on yet another trajectory
                traj = luck() * math.sqrt(0.5) + (traj if arm == "B" else 0.0)
            out[k] = (base[k] + (effect if arm == "B" else 0.0) + traj + shock + rng.gauss(0, req_sd)
                      - (drift * j / len(keys) if arm == "B" else 0.0))
        return out

    if sparkdash:
        k = keys[0]
        diff = effect + luckB[k] + rng.gauss(0, req_sd) * math.sqrt(2) + rng.gauss(0, boot_sd) * math.sqrt(2)
        return {"overall": {"est": diff, "ci": [diff - 0.01, diff + 0.01]}}
    if rigmark:
        # one prompt, 5 sequential repeats per arm, analysed the way a naive tool would: repeats as samples
        k = keys[0]
        a = [base[k] + rng.gauss(0, req_sd) for _ in range(5)]
        b = [base[k] + effect + luckB[k] + rng.gauss(0, req_sd) for _ in range(5)]
        diff = statistics.fmean(b) - statistics.fmean(a)
        se = math.sqrt(statistics.variance(a) / 5 + statistics.variance(b) / 5)
        tq = S.t_ppf(0.975, 8)
        return {"overall": {"est": diff, "ci": [diff - tq * se, diff + tq * se]}}
    A = [run("A", r) for r in range(runs)]
    B = [run("B", r) for r in range(runs)]
    return S.paired(A, B, cats, boot_prior_sd=boot_prior, n_boot=0, cluster=cluster,
                    shared_boot_sd=shared_prior if shared else None)


def scenario(name, sims, seed, **kw):
    rng = random.Random(seed)
    rows = {}
    for eff in (0.0, math.log(1.01), math.log(1.02)):
        cover = hit = 0
        widths = []
        for _ in range(sims):
            r = simulate(rng, effect=eff, **kw)
            lo, hi = r["overall"]["ci"]
            cover += lo <= eff <= hi
            hit += lo > 0 or hi < 0
            widths.append((hi - lo) / 2)
        rows[round(S.pct(eff))] = {"coverage": cover / sims, "reject": hit / sims,
                                   "ci_halfwidth_pct": statistics.median(widths) * 100}
    return {"name": name, "fp": rows[0]["reject"], "coverage0": rows[0]["coverage"],
            "power1": rows[1]["reject"], "power2": rows[2]["reject"], "halfwidth": rows[0]["ci_halfwidth_pct"],
            "coverage1": rows[1]["coverage"]}


def scenarios(quick=False):
    emp = empirical_luck()
    sd_emp = statistics.pstdev(emp)

    def make_emp(rng):
        return lambda: rng.choice(emp)

    def make_norm(rng, sd):
        return lambda: rng.gauss(0, sd)

    luck_rng = random.Random(99)
    # calibration is at 384 output tokens; scale the luck to other lengths by sqrt(384/L). Fewer tokens ->
    # more luck, so this is conservative for the 256/160-token modes and generous to RigMark's ~1000 tokens.
    def at(L, base):
        f = math.sqrt(384 / L)
        return lambda: base() * f
    E = make_emp(luck_rng)
    N = make_norm(luck_rng, sd_emp)
    L_small = make_norm(luck_rng, 0.004)      # trajectory-preserving (bit-exact) levers: timing noise only
    L_step = make_norm(luck_rng, 0.006)       # step time of a trajectory-changing step-cost lever (assumed SD)
    full = dict(n=73)
    il = dict(shared=True)                    # interleaved in one session (run --arm A=... --arm B=...)
    sep = dict(boot_sd=0.01, boot_prior=0.01) # separate sessions, session SD 1 % per arm (~1.4 % between arms)
    sc = [
        ("full 73x256, traj-changing, interleaved (empirical tails)", dict(full, luck=at(256, E), **il)),
        ("full 73x256, traj-changing, interleaved (normal)", dict(full, luck=at(256, N), **il)),
        ("full, trajectory-preserving, interleaved", dict(full, luck=L_small, **il)),
        ("full, traj-preserving, 2 interleaved sessions", dict(full, luck=L_small, runs=2, **il)),
        ("full, traj-preserving, 2 sessions, residual 0.3 %", dict(full, luck=L_small, runs=2, residual_sd=0.002, **il)),
        ("full, traj-changing, interleaved, nondet. T=0 (30 %)", dict(full, luck=at(256, E), nondet=0.3, **il)),
        ("full, traj-changing, interleaved, B-only drift 1 %", dict(full, luck=at(256, E), drift=0.01, **il)),
        ("--variants 4 (corr 0.3), traj-changing, interleaved", dict(full, luck=at(256, E), variants=4, **il)),
        # T>0: every (prompt, seed) is an independent trajectory in each arm (sampling); pairs per (prompt, seed)
        ("T>0, full, 1 seed per prompt, interleaved", dict(full, luck=at(256, E), **il)),
        ("T>0, full, 3 seeds per prompt, interleaved", dict(full, luck=at(256, E), variants=3, var_corr=0.0, **il)),
        ("T>0, quick 30, 3 seeds per prompt, interleaved", dict(n=30, luck=at(160, E), variants=3, var_corr=0.0, **il)),
        ("quick 30x160, traj-changing, interleaved", dict(n=30, luck=at(160, E), **il)),
        ("STEP TIME quick 30, step-cost lever, interleaved", dict(n=30, luck=L_step, **il)),
        ("STEP TIME full 73, step-cost lever, interleaved", dict(full, luck=L_step, **il)),
        ("cross-session, full, traj-changing, 1 session/arm", dict(full, luck=at(256, E), **sep)),
        ("cross-session, full, traj-preserving, 1 session/arm", dict(full, luck=L_small, **sep)),
        ("cross-session, traj-pres., true SD 1.4 % vs prior 1 %", dict(full, luck=L_small, boot_sd=0.014,
                                                                       boot_prior=0.01)),
        ("cross-session, traj-pres., 3 sessions/arm (ABABAB)", dict(full, luck=L_small, runs=3, **sep)),
        ("decset-like 24x384, traj-changing, interleaved", dict(n=24, luck=at(384, E), n_cats=3, **il)),
        ("RigMark-like 1 prompt x5 (~1000 tok), naive", dict(n=1, luck=at(1000, E), rigmark=True)),
        ("sparkDash-like 1 prompt/type (~512 tok), |diff|>1 %", dict(n=1, luck=at(512, E), sparkdash=True,
                                                                     boot_sd=0.001)),
    ]
    if quick:
        sc = [s for s in sc if "nondet" not in s[0] and "drift" not in s[0]]
    return sc, sd_emp


def trap_docs(rng, n=40, seeds=10, mode="b_only", short_tokens=4, slow=0.20, luck=0.02):
    """the outcome-dependent exclusion trap: per prompt, one seed of arm B is slow AND short; B's other seeds are
    faster by exactly as much, so the complete paired mean log effect is zero. mode: which arm's request of that seed
    is short (b_only, both) and how short (short_tokens; >= 8 is timeable at T>0)."""
    keep = math.log(1 - slow) / (seeds - 1)
    docs = []
    for arm in ("A", "B"):
        reqs = []
        for i in range(n):
            for s in range(seeds):
                trap = s == i % seeds
                lv = 0.0
                if arm == "B":
                    lv = math.log(1 - slow) if trap else -keep
                short = trap and (arm == "B" or mode == "both")
                reqs.append({"key": "p%02d@s%d" % (i, s), "id": "p%02d" % i, "category": "c%d" % (i % 4),
                             "kind": "measure", "ok": True, "valid_decode": not short,
                             "decode_tokens": short_tokens if short else 200, "tok_s": 40 * math.exp(lv + rng.gauss(0, luck)),
                             "sha256": "%s%d%d" % (arm, i, s), "repeat": 0, "seed": s, "boundary": "exact",
                             "ttft_s": 0.1, "t_wall": 2 * (i * seeds + s) + (arm == "B")})
        docs.append({"label": arm, "run_id": arm, "session_id": "trap", "prompt_set": {"version": "v", "sha256": "x"},
                     "settings": {"mode": "full", "temperature": 0.7, "seeds": "0-%d" % (seeds - 1), "max_tokens": 256,
                                  "effort": "low", "conc": 1, "prime": True},
                     "summary": {"requests": n * seeds, "failed": 0, "valid_decode": n * seeds}, "requests": reqs})
    return docs


def exclusion_trap(sims=20, seed=11):
    """fractions of publishable B FASTER, NOT PUBLISHABLE and WITHHELD verdicts in the trap (the complete-data effect
    is zero): a publishable B FASTER must never come out"""
    from . import compare
    args = argparse.Namespace(boot_prior=1.0, shared_prior=0.1, alt_prior=2.0, alpha=0.05, decide_on="tok",
                              tps_margin=2.0)
    rows = []
    for name, kw in (("B's slow seed fails the timing (4 tokens), B only", dict(mode="b_only", short_tokens=4)),
                     ("B's slow seed short but timeable (20 tokens)", dict(mode="b_only", short_tokens=20)),
                     ("the slow seed untimeable in both arms (4 tokens)", dict(mode="both", short_tokens=4)),
                     ("untimeable in both arms, 40 x 25 seeds (4 % dropped)", dict(mode="both", short_tokens=4,
                                                                                  seeds=25))):
        rng = random.Random(seed)
        verdicts = []
        for _ in range(sims):
            a, b = trap_docs(rng, **kw)
            verdicts.append(compare.analyse([a], [b], args)["verdict"])
        pub = [v for v in verdicts if not v.startswith(("WITHHELD", "NOT PUBLISHABLE"))]
        rows.append({"name": name, "b_faster": sum(v.endswith("B FASTER") for v in pub) / sims,
                     "not_publishable": sum(v.startswith("NOT PUBLISHABLE") for v in verdicts) / sims,
                     "withheld": sum(v.startswith("WITHHELD") for v in verdicts) / sims})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="speedbench selftest", description=__doc__)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    sc, sd_emp = scenarios(a.quick)
    print("empirical trajectory luck: per-prompt SD of the paired log ratio %.2f %% at 384 tokens "
          "(4 trajectory-changing fleet pairs, n=96); scaled to 256 tokens by sqrt(L) where noted" % (sd_emp * 100))
    print("%-58s %8s %8s %8s %8s %9s" % ("scenario", "cover@0", "FP@0", "pow@+1%", "pow@+2%", "CI +-%"))
    sys.stdout.flush()
    res = []
    for i, (name, kw) in enumerate(sc):
        r = scenario(name, a.sims, 1000 + i, **kw)
        res.append(r)
        print("%-58s %8.3f %8.3f %8.3f %8.3f %9.2f" % (name, r["coverage0"], r["fp"], r["power1"], r["power2"],
                                                         r["halfwidth"]), flush=True)
    print("\noutcome-dependent exclusion trap (T>0, 40 prompts; per prompt one seed of B is slow and short, the "
          "complete-data effect is zero; any dropped pair makes a verdict NOT PUBLISHABLE)")
    print("%-58s %12s %12s %10s" % ("scenario", "publ. FASTER", "NOT PUBL.", "WITHHELD"))
    for r in exclusion_trap(max(5, a.sims // 20)):
        print("%-58s %12.2f %12.2f %10.2f" % (r["name"], r["b_faster"], r["not_publishable"], r["withheld"]),
              flush=True)
        res.append(r)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
