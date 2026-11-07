"""Run a bot against the whole practice pool and print/save a compact summary.

    python dev/pool_eval.py agent.py --seeds 1-2 --tag v2 [--extra dev/versions/v1/agent_v1.py]
"""
import argparse
import json
import os
import statistics
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness  # noqa: E402

POOL = ["base_random", "base_calling_station", "base_always_pot", "base_always_half", "checkfold", "guide_example",
        "tag", "eq_tight", "eq_loose", "eq_aggro", "eq_potodds", "eqbot", "lag", "nit", "bluffer", "trapper",
        "diego_equity", "adaptive"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bot")
    p.add_argument("--seeds", default="1-2")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--tag", default=None)
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    opps = (POOL if a.only is None else a.only) + a.extra
    tasks = [(harness.resolve(a.bot), harness.resolve(o), s, a.deals) for o in opps for s in harness.parse_seeds(a.seeds)]
    t = time.time()
    res = []
    with Pool(os.cpu_count()) as pool:
        for r in pool.imap_unordered(harness.play_match, tasks):
            r.pop("paired", None)
            res.append(r)
            if r["errors_a"] or r["err_msg_a"]:
                print("ERROR", r["b"], r["seed"], (r["err_msg_a"] or "")[:2000], flush=True)
    by = {}
    for r in res:
        by.setdefault(r["b"], []).append(r)
    pts = sum(r["points_a"] for r in res)
    print(f"{os.path.basename(a.bot)} vs pool ({time.time() - t:.0f}s), {len(res)} matches:")
    for opp in sorted(by):
        rs = by[opp]
        m = statistics.mean(r["mean"] for r in rs)
        se = statistics.mean(r["stderr"] for r in rs)
        w = sum(r["verdict"] == "AGENT1_WIN" for r in rs)
        l = sum(r["verdict"] == "AGENT2_WIN" for r in rs)
        tmax = max(max(r["time_a"]) for r in rs)
        print(f"  {opp:26s} {m:+7.3f} (se {se:.3f})  W{w} C{len(rs) - w - l} L{l}  maxT {tmax:.2f}s")
    allm = statistics.mean(statistics.mean(r["mean"] for r in rs) for rs in by.values())
    print(f"  POINTS {pts:.1f}/{len(res)}   avg bb/deal over opponents {allm:+.3f}   "
          f"max half time {max(max(r['time_a']) for r in res):.2f}s")
    if a.tag:
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        with open(os.path.join(HERE, "results", f"pool_{a.tag}.json"), "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
