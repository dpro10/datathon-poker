"""Compare two saved pool runs opponent by opponent."""
import json, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
def load(tag):
    rs = json.load(open(os.path.join(HERE, "results", f"pool_{tag}.json")))
    by = {}
    for r in rs:
        by.setdefault(r["b"], {})[r["seed"]] = r["mean"]
    return by
a, b = load(sys.argv[1]), load(sys.argv[2])
diffs = []
for opp in sorted(set(a) & set(b)):
    seeds = sorted(set(a[opp]) & set(b[opp]))
    ma = statistics.mean(a[opp][s] for s in seeds); mb = statistics.mean(b[opp][s] for s in seeds)
    diffs.append(mb - ma)
    print(f"  {opp:26s} {sys.argv[1]} {ma:+7.3f}   {sys.argv[2]} {mb:+7.3f}   diff {mb - ma:+.3f}")
print(f"  mean diff {statistics.mean(diffs):+.3f} over {len(diffs)} opponents")
