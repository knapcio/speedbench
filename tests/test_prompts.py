import collections
import os
import subprocess
import sys
import unittest

from speedbench import prompts as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PromptSet(unittest.TestCase):
    def test_reproducible(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_prompts.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_shape(self):
        body, sha = P.load()
        ps = body["prompts"]
        self.assertEqual(len(ps), 73)
        self.assertEqual(len({p["id"] for p in ps}), 73)
        c = collections.Counter(p["category"] for p in ps)
        self.assertEqual(set(c), {"prose", "code", "structured", "agentic", "multilingual", "longctx"})
        self.assertEqual(len(P.select(body, "quick")), 30)
        self.assertEqual(sum(p["lang"] == "pl" for p in ps), 0)   # universal set: no Polish category
        for p in ps:
            self.assertTrue(p["messages"] and p["messages"][-1]["role"] in ("user", "tool"))

    def test_skip_tools(self):
        body, _ = P.load()
        self.assertFalse(any(p.get("tools") for p in P.select(body, "full", skip_tools=True)))


if __name__ == "__main__":
    unittest.main()
