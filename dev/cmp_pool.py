"""Compare per-opponent results of two bots from saved dev/pool_eval.py outputs.

Each side is NAME:FILE[@SECTION],FILE[@SECTION],... where SECTION is text from the "== ..."
header line that starts the block to read (for files holding several runs). Opponents found
on both sides are compared; the last line is the mean difference over opponents with its
standard error across opponents.

    python dev/cmp_pool.py "v8:runs/a.txt@v8 pool" "v9:runs/b.txt@share pool"
"""
import re, sys
import numpy as np

def parse(items):
    out = {}
    for item in items:
        p, _, section = item.partition("@")
        txt = open(p).read()
        if section:
            # take the block after the line containing `section`
            i = txt.find(section)
            if i < 0:
                continue
            txt = txt[i:]
            j = txt.find("\n==", 1)
            txt = txt if j < 0 else txt[:j]
        for m in re.finditer(r"^\s+(\S+\.py)\s+([+-]\d+\.\d+) \(se (\d+\.\d+)\)", txt, re.M):
            out.setdefault(m.group(1), []).append((float(m.group(2)), float(m.group(3))))
    return out

if __name__ == "__main__":
    # usage: cmp_pool.py "A:file1[@section],file2[@section]" "B:file3[@section]"
    specs = []
    for arg in sys.argv[1:3]:
        name, rest = arg.split(":", 1)
        specs.append((name, parse(rest.split(","))))
    (na, A), (nb, B) = specs
    diffs = []
    print(f"{'opponent':26s} {na:>8s} {nb:>8s}   diff")
    for o in sorted(set(A) & set(B)):
        a = np.mean([x for x, _ in A[o]]); b = np.mean([x for x, _ in B[o]])
        diffs.append(b - a)
        print(f"{o:26s} {a:+8.3f} {b:+8.3f} {b - a:+7.3f}")
    d = np.array(diffs)
    print(f"mean diff {d.mean():+.3f}  (se {d.std(ddof=1) / np.sqrt(len(d)):.3f}, {len(d)} opponents)")
