"""After one half vs an opponent, compare the bot's learned model with the opponent's true
behaviour (calibration counts). Prints, per class, the probability-weighted absolute error
of the learned P(action|bin) vs truth and vs the prior."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import harness
from game import AgentConfig, Game, make_deal, button_for_deal
from sdbot.model import CLASS_NAMES, prior

def run(opp, seed=5, deals=1000):
    cfg = AgentConfig(num_hands=deals)
    a = harness.Handle(os.path.join(ROOT, "agent.py"), cfg)
    b = harness.Handle(harness.resolve(opp), cfg)
    a.start(); b.start()
    bots = (a, b)
    for d in range(deals):
        g = Game(make_deal(seed, d), button_for_deal(d), d)
        while not g.is_terminal:
            s = g.to_act
            g.apply(bots[s].act(g.observation(s)))
        for s in (0, 1):
            bots[s].hand_end(g.hand_result(s))
    a.flush()
    return a.memory["model"]

cal = np.load(os.environ.get("CALIB", os.path.join(HERE, "results", "calib.npz")))
names = list(cal["names"])
pri = prior()
for opp in sys.argv[1:]:
    m = run(opp)
    learned = m.probs()
    true_c = cal["counts"][names.index(opp) if opp in names else 0]
    print(f"=== {opp}")
    tot_l = tot_p = tot_w = 0
    for c in range(17):
        w = true_c[c].sum(axis=1)            # how often each bin occurs
        if w.sum() < 20:
            continue
        truth = true_c[c] / np.maximum(true_c[c].sum(axis=1, keepdims=True), 1e-9)
        el = (np.abs(learned[c] - truth).sum(axis=1) * w).sum() / w.sum() / 2
        ep = (np.abs(pri[c] - truth).sum(axis=1) * w).sum() / w.sum() / 2
        tot_l += el * w.sum(); tot_p += ep * w.sum(); tot_w += w.sum()
        tm = (truth * w[:, None]).sum(0) / w.sum(); lm = (learned[c] * w[:, None]).sum(0) / w.sum()
        print(f"  {CLASS_NAMES[c]:15s} n={w.sum():5.0f} err learned {el:.3f} prior {ep:.3f} | true marg {np.round(tm,2)} learned {np.round(lm,2)}")
        if os.environ.get("SHAPE"):
            for b in range(5):
                print(f"        bin{b} w={w[b]:6.0f} true {np.round(truth[b],2)} learned {np.round(learned[c][b],2)}")
    print(f"  OVERALL weighted TV error: learned {tot_l/tot_w:.3f}  prior {tot_p/tot_w:.3f}")
