"""Blueprint leaf table -> sdbot/data/leaf_bp.npz

From blueprint self-play records (solver simulate): for every street that ended with chips
behind, y = (value of the rest of the hand) / pot - (all-in equity at that moment), i.e. what
future betting adds beyond equity when both players play the equilibrium. Tabulated by
street, stack-to-pot band, position (button = in position after the flop) and the hand-type
groups of both players, with sparse cells shrunk toward an additive row + column model.

The search uses it at street ends: leaf_k = pot * (e_k + g[street, spr, pos, my_group, group_k]).

    python build_leaf_table.py build/sim_a.f32 128 128 128
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(HERE, "build")
sys.path[:0] = [ROOT, HERE]
from fit_buckets import kmeans  # noqa: E402

NG = 32
SPR_EDGES = np.array([1.5, 3.0, 6.0, 12.0])
N0 = 200.0          # cell shrinkage (samples)
N0_RC = 2000.0      # row / column effect shrinkage: rare hand groups must not set extreme baselines
CLIP = 1.5          # |value beyond equity| cap, in pots


def groups_from_centroids(C, k=NG, seed=1):
    """Cluster bucket centroids into k groups (shape-aware), ordered weak -> strong."""
    G = kmeans(C.astype(np.float64), k, seed=seed)
    lab = ((C[:, None, :] - G[None, :, :]) ** 2).sum(2).argmin(1)
    return lab.astype(np.int16)


def preflop_groups(b0):
    d = np.load(os.path.join(ROOT, "sdbot", "data", "preflop.npz"))
    X, w = d["eq_bins"].astype(np.float64), d["counts"].astype(np.float64)
    lab = np.load(os.path.join(BUILD, f"pre_class_bucket_{b0}.npy")).astype(np.int64)
    C = np.zeros((b0, X.shape[1]))
    for b in range(b0):
        m = lab == b
        C[b] = np.average(X[m], axis=0, weights=w[m])
    order_key = C.mean(1)
    g = groups_from_centroids(C / 1.0)
    # relabel groups by mean equity so group index ~ strength
    gm = np.array([order_key[g == j].mean() if (g == j).any() else 0 for j in range(NG)])
    remap = np.argsort(np.argsort(gm))
    return remap[g].astype(np.int16), lab, X, w


def main(sim_path, b0, b1, b2):
    r = np.fromfile(sim_path, dtype=np.float32).reshape(-1, 8).astype(np.float64)
    s, P, S, bb0, bb1, e0, v0, _ = r.T
    s = s.astype(np.int64)
    gpre, cls_bucket, X, w = preflop_groups(b0)
    gflop = groups_from_centroids(np.fromfile(os.path.join(BUILD, f"cent_flop_{b1}.f32"), np.float32).reshape(b1, 8))
    gturn = groups_from_centroids(np.fromfile(os.path.join(BUILD, f"cent_turn_{b2}.f32"), np.float32).reshape(b2, 8))
    gmaps = [gpre, gflop, gturn]
    spr = np.searchsorted(SPR_EDGES, S / P, side="right")
    NS = len(SPR_EDGES) + 1
    g = np.zeros((3, NS, 2, NG, NG))
    n = np.zeros((3, NS, 2, NG, NG))
    for seat in (0, 1):
        pos = 1 if seat == 0 else 0                                    # seat 0 is the button
        me, op = (bb0, bb1) if seat == 0 else (bb1, bb0)
        e = e0 if seat == 0 else 1 - e0
        V = v0 if seat == 0 else P - v0
        y = V / P - e
        for st in range(3):
            m = s == st
            gm = gmaps[st][me[m].astype(np.int64)]
            go = gmaps[st][op[m].astype(np.int64)]
            np.add.at(g[st, :, pos], (spr[m], gm, go), y[m])
            np.add.at(n[st, :, pos], (spr[m], gm, go), 1)
    # shrink each (street, spr, pos) table toward mu + row + col
    out = np.zeros_like(g)
    for idx in np.ndindex(3, NS, 2):
        sm, cnt = g[idx], n[idx]
        tot = cnt.sum()
        if tot == 0:
            continue
        mu = sm.sum() / tot
        row = (sm.sum(1) - mu * cnt.sum(1)) / (cnt.sum(1) + N0_RC)
        col = (sm - (mu + row[:, None]) * cnt).sum(0) / (cnt.sum(0) + N0_RC)
        base = mu + row[:, None] + col[None, :]
        out[idx] = np.clip((sm + N0 * base) / (cnt + N0), -CLIP, CLIP)
    # preflop: the search works on 20 opponent percentile bins, so average over the hands in each bin
    import importlib
    pf = importlib.import_module("sdbot.preflop")
    pct = pf.PCT
    nb = pf.NB
    binj = np.minimum((pct * nb).astype(np.int64), nb - 1)
    grp_of_class = gpre[cls_bucket]
    W = np.zeros((nb, NG))
    np.add.at(W, (binj, grp_of_class), w)
    W /= W.sum(1, keepdims=True)
    pre_bins = np.einsum("sp mg, jg -> spmj".replace(" ", ""), out[0], W)    # (NS, 2, NG, nb)
    np.savez_compressed(os.path.join(ROOT, "sdbot", "data", "leaf_bp.npz"), g=out.astype(np.float32),
                        pre_bins=pre_bins.astype(np.float32), spr_edges=SPR_EDGES,
                        grp_pre=gpre, grp_flop=gflop, grp_turn=gturn, buckets=np.array([b0, b1, b2], np.int32))
    print(f"records {len(r)}; cells filled {[int((n[k] > 0).sum()) for k in range(3)]} of {NS * 2 * NG * NG} per street")
    for st in range(3):
        for pos in (0, 1):
            m = n[st, :, pos].sum()
            mean = (g[st, :, pos].sum()) / max(m, 1)
            print(f"  street {st} {'IP ' if pos else 'OOP'}: mean value beyond equity {mean:+.3f} pot; "
                  f"table spread (5-95%) {np.percentile(out[st, :, pos][n[st, :, pos] > 50], 5):+.2f}.."
                  f"{np.percentile(out[st, :, pos][n[st, :, pos] > 50], 95):+.2f}")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:5]))
