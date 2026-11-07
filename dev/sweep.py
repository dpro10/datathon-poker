"""Evaluate parameter variants: each variant plays a reference snapshot (self-play, paired seeds)
and a set of realistic pool bots. Usage:
    python dev/sweep.py --ref dev/versions/v2/agent_v2.py --variants '{"eta":0.0}' '{"eta":0.2}'
"""
import argparse, json, os, statistics, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PY = sys.executable
REAL = ["tag", "eq_tight", "nit", "trapper", "adaptive", "diego_equity", "eq_aggro", "lag"]

def run(variant, ref, seeds, pool_seeds, deals):
    env = dict(os.environ, SDBOT_PARAMS=variant, SDBOT_DEBUG="1")
    out = {}
    cmd = [PY, os.path.join(HERE, "pool_eval.py"), os.path.join(ROOT, "agent.py"), "--seeds", seeds, "--deals", str(deals),
           "--only", "--extra", ref]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sp = [l for l in r.stdout.splitlines() if os.path.basename(ref) in l]
    cmd2 = [PY, os.path.join(HERE, "pool_eval.py"), os.path.join(ROOT, "agent.py"), "--seeds", pool_seeds, "--deals", str(deals),
            "--only"] + REAL
    r2 = subprocess.run(cmd2, env=env, capture_output=True, text=True)
    tot = [l for l in r2.stdout.splitlines() if "POINTS" in l]
    errs = [l for l in (r.stdout + r2.stdout).splitlines() if l.startswith("ERROR")]
    return (sp[0].strip() if sp else r.stdout[-500:] + r.stderr[-2000:]), (tot[0].strip() if tot else r2.stdout[-500:] + r2.stderr[-2000:]), errs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--seeds", default="1-8")
    p.add_argument("--pool-seeds", default="1")
    p.add_argument("--deals", type=int, default=1000)
    a = p.parse_args()
    for v in a.variants:
        sp, pool, errs = run(v, a.ref, a.seeds, a.pool_seeds, a.deals)
        print(f"{v}\n   selfplay: {sp}\n   realpool: {pool}", flush=True)
        for e in errs[:3]:
            print("   ", e[:500])

if __name__ == "__main__":
    main()
