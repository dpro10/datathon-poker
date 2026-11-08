"""Full evaluation of a parameter variant of agent.py: the practice pool, earlier versions,
and any extra opponents. Prints one summary block.

    python dev/suite.py '{"kl_kappa":0.1}' --seeds 1-2 --extra dev/versions/bp1/agent_bp1.py
"""
import argparse, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", nargs="?", default="{}")
    p.add_argument("--seeds", default="1-2")
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    env = dict(os.environ, SDBOT_PARAMS=a.variant, SDBOT_DEBUG="1")
    cmd = [sys.executable, os.path.join(HERE, "pool_eval.py"), os.path.join(ROOT, "agent.py"), "--seeds", a.seeds,
           "--extra", *a.extra] + (["--tag", a.tag] if a.tag else [])
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(f"== {a.variant}")
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith("  ") or l.startswith("ERROR")))
    if r.returncode:
        print(r.stderr[-3000:])

if __name__ == "__main__":
    main()
