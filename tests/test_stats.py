import math
import random
import unittest

from speedbench import stats as S


class Distributions(unittest.TestCase):
    def test_t_quantiles(self):
        for df, q in ((1, 12.706), (5, 2.571), (10, 2.228), (30, 2.042), (120, 1.980)):
            self.assertAlmostEqual(S.t_ppf(0.975, df), q, places=2)
        self.assertAlmostEqual(S.t_ppf(0.975, math.inf), 1.95996, places=4)
        self.assertAlmostEqual(S.t_cdf(0.0, 7), 0.5, places=12)


class Paired(unittest.TestCase):
    def make(self, n=40, effect=0.02, sd=0.03, runs=1, seed=0):
        rng = random.Random(seed)
        cats = {"k%02d" % i: ("x" if i % 2 else "y") for i in range(n)}
        base = {k: rng.gauss(3.7, 0.1) for k in cats}
        luck = {k: rng.gauss(0, sd) for k in cats}
        A = [{k: base[k] + rng.gauss(0, 0.002) for k in cats} for _ in range(runs)]
        B = [{k: base[k] + effect + luck[k] + rng.gauss(0, 0.002) for k in cats} for _ in range(runs)]
        return A, B, cats

    def test_recovers_effect(self):
        A, B, cats = self.make()
        r = S.paired(A, B, cats, n_boot=200)
        self.assertLess(abs(r["overall"]["est"] - 0.02), 0.015)
        self.assertLess(r["overall"]["ci"][0], r["overall"]["est"])
        self.assertEqual(r["n_prompts"], 40)
        self.assertEqual(set(r["categories"]), {"x", "y"})

    def test_pairwise_exclusion_and_clusters(self):
        A, B, cats = self.make()
        del B[0]["k03"]                                  # failed in one arm -> dropped from both
        r = S.paired(A, B, cats, n_boot=0)
        self.assertEqual(r["excluded"], ["k03"])
        # two keys of one prompt (e.g. two seeds) form one cluster, not two samples
        cl = {k: ("same" if k in ("k00", "k02") else k) for k in cats}
        r2 = S.paired(A, B, cats, n_boot=0, cluster=cl)
        self.assertEqual(r2["n_prompts"], 38)

    def test_boot_term_from_runs(self):
        A, B, cats = self.make(runs=3)
        r = S.paired(A, B, cats, n_boot=0)
        self.assertEqual(r["boot"]["A"]["source"], "anova+prior")
        self.assertEqual(r["boot"]["A"]["df"], 4 + 2)
        r1 = S.paired(A[:1], B[:1], cats, n_boot=0)
        self.assertEqual(r1["boot"]["A"]["source"], "prior")

    def test_coverage_under_trajectory_luck(self):
        """~95 % of 95 % CIs cover the true effect (heavy-tailed luck, boot shocks, one run per arm)"""
        rng = random.Random(7)
        cover = 0
        sims = 300
        for i in range(sims):
            n = 60
            cats = {"k%02d" % j: "c%d" % (j % 5) for j in range(n)}
            shockA, shockB = rng.gauss(0, 0.0025), rng.gauss(0, 0.0025)
            A = [{k: 3.8 + shockA for k in cats}]
            B = [{k: 3.8 + 0.01 + shockB + (rng.gauss(0, 0.04) if rng.random() > .05 else rng.gauss(0, .12))
                  for k in cats}]
            lo, hi = S.paired(A, B, cats, n_boot=0, boot_prior_sd=0.0025)["overall"]["ci"]
            cover += lo <= 0.01 <= hi
        self.assertGreater(cover / sims, 0.91)
        self.assertLess(cover / sims, 0.99)

    def test_verdict_labels(self):
        A, B, cats = self.make(effect=0.08, sd=0.01)
        self.assertEqual(S.verdict(S.paired(A, B, cats, n_boot=0)), "B FASTER")
        self.assertEqual(S.verdict(S.paired(B, A, cats, n_boot=0)), "B SLOWER")


if __name__ == "__main__":
    unittest.main()
