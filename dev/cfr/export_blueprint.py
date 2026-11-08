"""Pack a trained strategy into sdbot/data/blueprint.npz (see sdbot/blueprint.py)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
BUILD = os.path.join(HERE, "build")

def main(prefix, b0, b1, b2, b3, out=os.path.join(ROOT, "sdbot", "data", "blueprint.npz")):
    t = np.load(os.path.join(HERE, "tree.npz"))
    B = [b0, b1, b2, b3]
    kind, street = t["kind"], t["street"]
    offset = np.full(len(kind), -1, dtype=np.int64)
    ni = 0
    for n in range(len(kind)):
        if kind[n] == 0:
            offset[n] = ni
            ni += B[street[n]]
    S = np.fromfile(prefix + ".strategy.f32", dtype=np.float32).reshape(-1, 4)
    assert len(S) == ni, (len(S), ni)
    q = np.clip(np.round(S * 255), 0, 255).astype(np.uint8)
    q[S[:, 0] < 0] = 255                           # never reached
    np.savez_compressed(out, child=t["child"].astype(np.int32), street=street.astype(np.int8), offset=offset,
                        strat=q, buckets=np.array(B, np.int32),
                        pre=np.load(os.path.join(BUILD, f"pre_class_bucket_{b0}.npy")).astype(np.uint16),
                        cflop=np.fromfile(os.path.join(BUILD, f"cent_flop_{b1}.f32"), np.float32).reshape(b1, 8),
                        cturn=np.fromfile(os.path.join(BUILD, f"cent_turn_{b2}.f32"), np.float32).reshape(b2, 8))
    reached = (S[:, 0] >= 0).mean()
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB), {ni} info sets, {reached:.1%} reached")

if __name__ == "__main__":
    # python export_blueprint.py build/bp_a 128 128 128 64 [out.npz]
    main(sys.argv[1], *(int(x) for x in sys.argv[2:6]), *sys.argv[6:7])
