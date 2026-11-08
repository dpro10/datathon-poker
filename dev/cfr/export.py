"""Export everything the C solver needs (dev/cfr/build/*.bin) and fit the preflop clusters.

tables.bin : NF, FL (45*165 int16 each), PAIR_IDX (81 uint8), TRIP_IDX (729 uint8)
pre.bin    : uint16[376992] preflop bucket of every hand (colex index)
tree.bin   : header (n_nodes) then player, street, kind, win (int32[n]), child (int32[n*4]), contrib (int32[n*2])
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from sdbot import evaluator as ev  # noqa: E402
from sdbot import preflop as pf  # noqa: E402

BUILD = os.path.join(HERE, "build")


def kmeans(X, k, w=None, iters=60, seed=0):
    """Weighted k-means (k-means++ init, Lloyd iterations)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    w = np.ones(n) if w is None else w.astype(np.float64)
    c = [X[rng.choice(n, p=w / w.sum())]]
    d2 = ((X - c[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = w * d2
        c.append(X[rng.choice(n, p=p / p.sum())])
        d2 = np.minimum(d2, ((X - c[-1]) ** 2).sum(1))
    C = np.array(c)
    for _ in range(iters):
        dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2) if n * k < 4e7 else None
        if dist is None:
            lab = np.empty(n, dtype=np.int64)
            for s in range(0, n, 20000):
                lab[s:s + 20000] = ((X[s:s + 20000, None, :] - C[None]) ** 2).sum(2).argmin(1)
        else:
            lab = dist.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = np.average(X[m], axis=0, weights=w[m])
            else:  # re-seed an empty cluster at the worst-fit point
                C[j] = X[rng.integers(n)]
    order = np.argsort(C.mean(1))      # sort clusters weak -> strong (mean equity)
    C = C[order]
    lab = np.argsort(order)[lab]
    return C, lab


def main(b0=128):
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, "tables.bin"), "wb") as f:
        f.write(ev.NF.astype(np.int16).tobytes())
        f.write(ev.FL.astype(np.int16).tobytes())
        f.write(ev.PAIR_IDX.astype(np.uint8).tobytes())
        f.write(ev.TRIP_IDX.astype(np.uint8).tobytes())
    # preflop: cluster the 19,998 suit classes by their equity profile vs 20 strength slices
    X = pf.EQ_BINS
    w = np.load(os.path.join(ROOT, "sdbot", "data", "preflop.npz"))["counts"].astype(np.float64)
    C, lab = kmeans(X, b0, w)
    pre = lab[pf.CLS_OF_HAND].astype(np.uint16)
    pre.tofile(os.path.join(BUILD, "pre.bin"))                       # what the solver loads
    pre.tofile(os.path.join(BUILD, f"pre_{b0}.bin"))
    np.save(os.path.join(BUILD, f"pre_class_bucket_{b0}.npy"), lab.astype(np.uint16))
    print(f"preflop: {b0} clusters; class-size-weighted cluster sizes min {np.bincount(lab, weights=w).min():.0f} "
          f"max {np.bincount(lab, weights=w).max():.0f}")
    t = np.load(os.path.join(HERE, "tree.npz"))
    n = len(t["player"])
    with open(os.path.join(BUILD, "tree.bin"), "wb") as f:
        f.write(np.array([n], np.int32).tobytes())
        for k in ("player", "street", "kind", "win"):
            f.write(t[k].astype(np.int32).tobytes())
        f.write(t["child"].astype(np.int32).tobytes())
        f.write(t["contrib"].astype(np.int32).tobytes())
    print("exported tables, preflop buckets and tree with", n, "nodes")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 128)
