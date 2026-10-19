"""python3 -m speedbench {run,compare,load,longctx,report,registry,mock,selftest} ..."""
import sys

USAGE = """usage: python3 -m speedbench <command> [args]

commands:
  run       measure one endpoint (or two, interleaved) over the prompt set -> JSON result
  compare   paired A/B analysis of result files, with confidence interval, decomposition and step-time rule
  load      concurrency levels (aggregate and per-stream tok/s, TTFT) and the agentic mix
  longctx   prefill/TTFT curve and decode speed at long contexts, cold and prefix-cached
  report    print one result file of any mode, with the per-category speed decomposition
  registry  append runs to results/registry.jsonl and write the README results tables
  mock      start the mock OpenAI-compatible server (simulated speculative decoding)
  selftest  Monte Carlo check that the confidence intervals hold under trajectory luck
"""


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        return
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        from .run import main as m
    elif cmd == "compare":
        from .compare import main as m
    elif cmd == "load":
        from .load import main as m
    elif cmd == "longctx":
        from .longctx import main as m
    elif cmd == "report":
        from .report import main as m
    elif cmd == "registry":
        from .registry import main as m
    elif cmd == "mock":
        from .mock_server import main as m
    elif cmd == "selftest":
        from .selftest import main as m
    else:
        print(USAGE)
        sys.exit(2)
    m(argv)


if __name__ == "__main__":
    main()
