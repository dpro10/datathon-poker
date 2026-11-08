"""Fit flop/turn bucket centroids: k-means on the solver's 8-bin final-strength histograms."""
import os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")


def kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    x2 = (X ** 2).sum(1)
    C = [X[rng.integers(n)]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):                        # k-means++ seeding
        C.append(X[rng.choice(n, p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for it in range(iters):
        dist = x2[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        lab = dist.argmin(1)
        cnt = np.bincount(lab, minlength=k)
        newC = np.zeros_like(C)
        np.add.at(newC, lab, X)
        empty = cnt == 0
        newC[~empty] /= cnt[~empty, None]
        newC[empty] = X[rng.integers(n, size=empty.sum())]
        shift = np.abs(newC - C).max()
        C = newC
        if shift < 1e-6:
            break
    centers = np.arange(8) + 0.5
    order = np.argsort(C @ centers)              # weak -> strong by mean strength
    return C[order].astype(np.float32)


def main(n=300000, b1=128, b2=128):
    for street, b in (("flop", b1), ("turn", b2)):
        path = os.path.join(BUILD, f"feat_{street}.f32")
        if not os.path.exists(path):
            subprocess.run([os.path.join(HERE, "solver"), "features", street, str(n), path], check=True, cwd=HERE)
        X = np.fromfile(path, dtype=np.float32).reshape(-1, 8).astype(np.float64)
        C = kmeans(X, b)
        C.tofile(os.path.join(BUILD, f"cent_{street}_{b}.f32"))
        lab = ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1) if len(X) * b < 2e7 else None
        print(f"{street}: {len(X)} samples -> {b} centroids; mean strength range "
              f"{(C @ (np.arange(8) + 0.5) / 8).min():.3f}..{(C @ (np.arange(8) + 0.5) / 8).max():.3f}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
