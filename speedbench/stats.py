"""Paired statistics for decode-speed A/B (stdlib only).

Model for the paired per-prompt log speed ratio of arm B over arm A:

    y[arm][run][i] = m[arm][i] + b[arm][run] + e      (b = boot/run shock shared by all prompts of a run)
    d_i            = mean_runs y[B][.][i] - mean_runs y[A][.][i]

The estimate is a fixed-weight mean of d_i (weights = prompt-set category weights, i.e. equal per prompt).
Its variance has a prompt part (stratified sample variance of d_i, which already contains trajectory luck and
request noise) and a boot part sigma_b^2/R per arm, with sigma_b from a two-way ANOVA of each arm's runs (needs
>= 2 runs) or, with one run, from a declared prior. The confidence interval and p-value use Student t with
Welch-Satterthwaite degrees of freedom, so one model drives both.
"""
import math
import random
import statistics

Z975 = 1.959963984540054
Z80 = 0.8416212335729143


# ------------------------------------------------------------------------------------------ distributions
def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (Numerical Recipes, modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    if df is None or math.isinf(df) or df > 1e6:
        return statistics.NormalDist().cdf(t)
    x = df / (df + t * t)
    p = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1 - p if t > 0 else p


def t_ppf(q, df):
    if df is None or math.isinf(df) or df > 1e6:
        return statistics.NormalDist().inv_cdf(q)
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------------------------------------ helpers
def pct(x):
    """log ratio -> percent change"""
    return (math.exp(x) - 1) * 100


def quantile(v, q):
    v = sorted(v)
    if not v:
        return None
    pos = (len(v) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def hodges_lehmann(d):
    w = [(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))]
    return statistics.median(w)


def winsorized_mean(d, k=3.0):
    med = statistics.median(d)
    mad = statistics.median(abs(x - med) for x in d) * 1.4826
    if mad == 0:
        return statistics.fmean(d), []
    lo, hi = med - k * mad, med + k * mad
    clipped = [i for i, x in enumerate(d) if x < lo or x > hi]
    return statistics.fmean(min(max(x, lo), hi) for x in d), clipped


def anova_boot(Y):
    """Y: runs x prompts matrix (complete). Returns (sigma_b^2, df_b, ms_resid) for y = m_i + b_r + e."""
    R, N = len(Y), len(Y[0])
    if R < 2 or N < 2:
        return None, None, None
    g = statistics.fmean(v for row in Y for v in row)
    pm = [statistics.fmean(Y[r][i] for r in range(R)) for i in range(N)]
    rm = [statistics.fmean(row) for row in Y]
    ms_run = N * sum((m - g) ** 2 for m in rm) / (R - 1)
    ms_e = sum((Y[r][i] - pm[i] - rm[r] + g) ** 2 for r in range(R) for i in range(N)) / ((R - 1) * (N - 1))
    return max(0.0, (ms_run - ms_e) / N), R - 1, ms_e


# ------------------------------------------------------------------------------------------ main analysis
def paired(A, B, cats, boot_prior_sd=0.01, alpha=0.05, n_boot=4000, seed=1, weights=None, cluster=None,
           prior_df=4, shared_boot_sd=None, partial=False):
    """A, B: lists of runs; each run is {key: log value}. cats: {key: category}.

    Keys are grouped per cluster (prompt id: seeds, trajectory variants and repeats of one prompt are not
    independent samples). A cluster is used only if ALL its planned keys are valid in every run of both arms
    (dropping a failed variant would bias the prompt's mean); its keys are averaged inside each run. Category
    weights are frozen at the planned prompt counts unless `weights` is given, so exclusions do not reweight.
    Returns the overall estimate and per-category estimates on the log scale.

    partial=True (T>0 with seeds): a cluster is used with those of its keys that are valid in every run of both
    arms; a (prompt, seed) pair missing in either arm is dropped from both. At T>0 every seed is an independent
    draw of the same distribution, so dropping a pair does not shift the prompt's expected difference the way a
    failed T=0 variant would (variants differ systematically). Dropped pairs are counted (pairs_dropped).
    """
    cl = cluster or {}
    planned = {}
    for k in cats:
        planned.setdefault(cl.get(k, k), []).append(k)
    if partial:
        clusters = {c: sorted(k for k in ks if all(k in r for r in A) and all(k in r for r in B))
                    for c, ks in planned.items()}
        clusters = {c: ks for c, ks in clusters.items() if ks}
    else:
        clusters = {c: sorted(ks) for c, ks in planned.items()
                    if all(k in r for k in ks for r in A) and all(k in r for k in ks for r in B)}
    pairs_planned = sum(len(ks) for ks in planned.values())
    pairs_used = sum(len(ks) for ks in clusters.values())
    excluded = sorted(set(planned) - set(clusters))
    planned_counts = {}
    for c, ks in planned.items():
        planned_counts[cats[ks[0]]] = planned_counts.get(cats[ks[0]], 0) + 1
    if len(clusters) < 3:
        raise ValueError("fewer than 3 prompts valid in every run of both arms")
    A = [{c: statistics.fmean(r[k] for k in ks) for c, ks in clusters.items()} for r in A]
    B = [{c: statistics.fmean(r[k] for k in ks) for c, ks in clusters.items()} for r in B]
    cats = {c: cats[ks[0]] for c, ks in clusters.items()}
    keys = sorted(clusters)
    YA = [[r[k] for k in keys] for r in A]
    YB = [[r[k] for k in keys] for r in B]
    mA = [statistics.fmean(YA[r][i] for r in range(len(A))) for i in range(len(keys))]
    mB = [statistics.fmean(YB[r][i] for r in range(len(B))) for i in range(len(keys))]
    d = [b - a for a, b in zip(mA, mB)]
    boot = {}
    shared_est = None
    if shared_boot_sd is not None and len(YA) >= 2 and len(YA) == len(YB):
        # several interleaved sessions: the session shock cancels inside each session, but the paired difference
        # can still move from session to session (order effects, a lever whose cost depends on host state).
        # Estimate that from the data: two-way ANOVA (session x prompt) of the per-session differences, pooled
        # with the residual prior. Luck is identical in every session (same T=0 text), so it sits in the prompt
        # term and is not counted again here.
        D = [[YB[r][i] - YA[r][i] for i in range(len(keys))] for r in range(len(YA))]
        s2, df_s, _ = anova_boot(D)
        pooled = (prior_df * shared_boot_sd ** 2 + df_s * s2) / (prior_df + df_s)
        shared_est = {"sigma_session_anova": math.sqrt(s2), "anova_df": df_s, "sigma_pooled": math.sqrt(pooled)}
    for name, Y in (("A", YA), ("B", YB)):
        if shared_boot_sd is not None:
            # both arms ran interleaved on the same boots: the boot shock cancels in d_i; what remains is a
            # residual (order/drift) term for the difference, split over the two arms (sigma/sqrt(2) each)
            if shared_est:
                boot[name] = {"runs": len(Y), "sigma_b": shared_est["sigma_pooled"] / math.sqrt(2),
                              "df": (prior_df + shared_est["anova_df"]) / 2.0,
                              "source": "shared sessions: anova+prior", **shared_est}
            else:
                boot[name] = {"runs": len(Y), "sigma_b": shared_boot_sd / math.sqrt(2), "df": math.inf,
                              "source": "shared boots (interleaved), residual prior"}
            continue
        s2, df, mse = anova_boot(Y)
        if s2 is None:
            boot[name] = {"runs": len(Y), "sigma_b": boot_prior_sd, "df": math.inf, "source": "prior"}
        else:
            # a 1-3 df ANOVA estimate is often ~0; pool it with the prior as `prior_df` pseudo-observations
            pooled = (prior_df * boot_prior_sd ** 2 + df * s2) / (prior_df + df)
            boot[name] = {"runs": len(Y), "sigma_b": math.sqrt(pooled), "df": prior_df + df,
                          "source": "anova+prior", "sigma_b_anova": math.sqrt(s2), "anova_df": df,
                          "sigma_resid": math.sqrt(mse)}

    by_cat = {}
    for i, k in enumerate(keys):
        by_cat.setdefault(cats[k], []).append(d[i])
    multi = [v for v in by_cat.values() if len(v) >= 2]
    pooled_var = (sum(sum((x - statistics.fmean(v)) ** 2 for x in v) for v in multi)
                  / max(1, sum(len(v) - 1 for v in multi))) if multi else 0.0

    def estimate(idx, w_by_cat):
        groups = {}
        for i in idx:
            groups.setdefault(cats[keys[i]], []).append(d[i])
        tot = sum(w_by_cat[c] for c in groups)
        w = {c: w_by_cat[c] / tot for c in groups}
        est = sum(w[c] * statistics.fmean(v) for c, v in groups.items())
        vp, dfp_terms = 0.0, []
        for c, v in groups.items():
            if len(v) >= 2:
                term = w[c] ** 2 * statistics.variance(v) / len(v)
                vp += term
                dfp_terms.append((term, len(v) - 1))
            else:                       # a single prompt carries no variance estimate: borrow the pooled one
                vp += w[c] ** 2 * pooled_var
        vb_terms = [(boot[a]["sigma_b"] ** 2 / boot[a]["runs"], boot[a]["df"]) for a in ("A", "B")]
        var = vp + sum(t for t, _ in vb_terms)
        den = sum(t * t / df for t, df in dfp_terms + vb_terms if df and not math.isinf(df) and t > 0)
        df = (var * var / den) if den > 0 else math.inf
        se = math.sqrt(var)
        tq = t_ppf(1 - alpha / 2, df)
        p = 2 * (1 - t_cdf(abs(est) / se, df)) if se > 0 else 0.0
        return {"n": len(idx), "est": est, "se": se, "se_prompt": math.sqrt(vp), "df": df,
                "ci": [est - tq * se, est + tq * se], "p": p, "groups": groups}

    present = {}
    for k in keys:
        present[cats[k]] = present.get(cats[k], 0) + 1
    w_by_cat = dict(weights) if weights else {c: planned_counts[c] for c in present}
    overall = estimate(range(len(keys)), w_by_cat)
    per_cat = {}
    for c in sorted(present):
        per_cat[c] = estimate([i for i, k in enumerate(keys) if cats[k] == c], {c: 1.0})
        per_cat[c].pop("groups")
    groups = overall.pop("groups")
    # pooled within-category SD of d_i (the per-prompt noise that sets the sample size)
    ss = sum(sum((x - statistics.fmean(v)) ** 2 for x in v) for v in groups.values())
    dof = sum(len(v) - 1 for v in groups.values())
    sd_d = math.sqrt(ss / dof) if dof > 0 else float("nan")
    # power: MDE at 80 % power, and prompts needed for a 1 % MDE with the same per-prompt SD when every arm
    # has R runs (at deterministic T=0 extra runs shrink only the boot term, not trajectory luck)
    zsum = t_ppf(1 - alpha / 2, overall["df"]) + t_ppf(0.8, overall["df"])
    mde = zsum * overall["se"]
    target = math.log(1.01) / zsum
    need = {}
    for R in (1, 2, 4):
        vb = sum(boot[a]["sigma_b"] ** 2 / R for a in ("A", "B"))
        need[R] = None if target ** 2 <= vb else math.ceil(sd_d ** 2 / (target ** 2 - vb))
    # sensitivity lines (descriptive, not the verdict)
    wm, clipped = winsorized_mean(d)
    hl = hodges_lehmann(d)
    loo = max(abs(statistics.fmean(d[:i] + d[i + 1:]) - statistics.fmean(d)) for i in range(len(d)))
    rng = random.Random(seed)
    bs = []
    idx_by_cat = {}
    for i, k in enumerate(keys):
        idx_by_cat.setdefault(cats[k], []).append(i)
    wsum = sum(w_by_cat[c] for c in idx_by_cat)
    for _ in range(n_boot):
        e = 0.0
        for c, idx in idx_by_cat.items():
            e += w_by_cat[c] / wsum * statistics.fmean(d[rng.choice(idx)] for _ in idx)
        bs.append(e)
    order = sorted(range(len(d)), key=lambda i: d[i])
    return {
        "overall": overall, "categories": per_cat, "boot": boot, "n_prompts": len(keys),
        "excluded": excluded, "pairs_planned": pairs_planned, "pairs_dropped": pairs_planned - pairs_used,
        "used_keys": sorted(k for ks in clusters.values() for k in ks), "sd_d": sd_d, "mde80": mde, "prompts_for_1pct": need,
        "bootstrap_ci_prompt_only": [quantile(bs, alpha / 2), quantile(bs, 1 - alpha / 2)],
        "median_d": statistics.median(d), "hodges_lehmann": hl, "winsorized_mean": wm,
        "winsorized": [keys[i] for i in clipped], "loo_max_shift": loo,
        "worst": [(keys[i], d[i]) for i in order[:3]], "best": [(keys[i], d[i]) for i in order[-3:][::-1]],
        "per_prompt": {k: d[i] for i, k in enumerate(keys)},
    }


def verdict(res, alpha=0.05):
    o = res["overall"]
    lo, hi = o["ci"]
    if lo > 0:
        return "B FASTER"
    if hi < 0:
        return "B SLOWER"
    return "NO DETECTABLE DIFFERENCE (MDE80 %.1f %%)" % pct(res["mde80"])


def _binom_two_sided(k, n):
    """two-sided exact binomial p-value for k of n at p = 0.5"""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def exclusion_check(A, B, cats, res, paired_kw):
    """Is the verdict robust to the requests that were excluded (failed, or too short to time)?

    Exclusions can depend on the outcome: if the slow trajectories of one arm are also the short ones, dropping them
    makes that arm look faster although nothing changed. Two checks:
      1. attribution: for every dropped key, which arm lacks a valid value (A only, B only, both); a two-sided exact
         binomial test of A-only vs B-only drops. Drops concentrated in one arm are associated with the arm.
      2. worst-case bound: every dropped value is imputed at the least favourable end of what that arm produced on the
         prompt's other keys, or at the least favourable paired difference observed anywhere, whichever is worse, and
         the CI is recomputed. If it then includes zero, the verdict depends on the exclusions. Values more extreme
         than anything observed cannot be bounded: hence also the cap on dropped pairs.
    A, B: runs as given to paired(); res: the paired() result; paired_kw: its other arguments."""
    used = set(res["used_keys"])
    dropped = [k for k in cats if k not in used]
    okA = {k for k in cats if all(k in r for r in A)}
    okB = {k for k in cats if all(k in r for r in B)}
    a_only = [k for k in dropped if k not in okA and k in okB]
    b_only = [k for k in dropped if k in okA and k not in okB]
    both = [k for k in dropped if k not in okA and k not in okB]
    out = {"dropped": len(dropped), "a_only": len(a_only), "b_only": len(b_only), "both": len(both),
           "p_imbalance": _binom_two_sided(len(a_only), len(a_only) + len(b_only)), "bound_ci": None}
    lo, hi = res["overall"]["ci"]
    if dropped and (lo > 0 or hi < 0):
        faster = lo > 0                     # the verdict to challenge: B FASTER (else B SLOWER)
        out["direction"] = 1 if faster else -1
        cl = paired_kw.get("cluster") or {}
        mA = {k: statistics.fmean(r[k] for r in A) for k in okA}
        mB = {k: statistics.fmean(r[k] for r in B) for k in okB}
        d = [mB[k] - mA[k] for k in used]
        d_imp = min(d) if faster else max(d)
        # per prompt, the range of the values each arm produced on its other keys (seeds, turns): a dropped value
        # is imputed at the end of that range least favourable to the verdict, or at the least favourable paired
        # difference observed anywhere, whichever is worse
        byA, byB = {}, {}
        for k in mA:
            byA.setdefault(cl.get(k, k), []).append(mA[k])
        for k in mB:
            byB.setdefault(cl.get(k, k), []).append(mB[k])
        pick_b, pick_a = (min, max) if faster else (max, min)
        worse = min if faster else max
        A2 = [dict(r) for r in A]
        B2 = [dict(r) for r in B]
        for k in dropped:
            c = cl.get(k, k)
            if k in b_only:
                va = mA[k]
                vb = worse(va + d_imp, pick_b(byB[c])) if byB.get(c) else va + d_imp
            elif k in a_only:
                vb = mB[k]
                va = vb - d_imp
                if byA.get(c):
                    va = pick_a(va, pick_a(byA[c]))
            elif k in both:
                va = pick_a(byA[c]) if byA.get(c) else 0.0
                vb = worse(va + d_imp, pick_b(byB[c])) if byB.get(c) else va + d_imp
            else:
                continue
            for r in A2:
                r[k] = va
            for r in B2:
                r[k] = vb
        try:
            r2 = paired(A2, B2, cats, **dict(paired_kw, n_boot=0))
            out["bound_ci"] = r2["overall"]["ci"]
            out["bound_est"] = r2["overall"]["est"]
        except ValueError:
            pass
    return out


def exclusion_rule(check, planned_pairs):
    """None if the verdict may stand, else the reason to withhold it"""
    single = check["a_only"] + check["b_only"]
    if single >= 5 and check["p_imbalance"] < 0.10:
        return ("exclusions concentrated in one arm (A only %d, B only %d, p=%.2g): when an arm's failures or short "
                "outputs are also its slow (or fast) ones, dropping them fakes an effect" % (
                    check["a_only"], check["b_only"], check["p_imbalance"]))
    ci = check.get("bound_ci")
    if ci is not None:
        # the verdict must keep its direction: a sensitivity CI that includes zero or lies on the other side
        # (a reversal) means the verdict depends on the exclusions
        kept = ci[0] > 0 if check.get("direction", 1) > 0 else ci[1] < 0
        if not kept:
            return ("not robust to exclusions: with the %d dropped pairs imputed at the least favourable observed "
                    "values the CI is [%+.2f, %+.2f] %%, which %s" % (
                        check["dropped"], pct(ci[0]), pct(ci[1]),
                        "includes zero" if ci[0] <= 0 <= ci[1] else "reverses the verdict"))
    return None
