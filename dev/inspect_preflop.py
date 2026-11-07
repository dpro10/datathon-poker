"""Show what the bot does preflop by hand percentile (with the prior opponent model)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from game import Game, make_deal, Action, AgentConfig, Observation
from sdbot import brain, preflop as pf

memory = {}
brain.new_half(AgentConfig(), memory)
rng = np.random.default_rng(0)
# collect hands by percentile
rows = []
for i in range(4000):
    memory["time"] = 0.0
    g = Game(make_deal("insp", i), 0, i)
    obs = g.observation(0)  # SB first decision
    p = pf.pct_of(obs.hole_cards)
    a = brain.choose_action(obs, memory)
    memory["hand"] = None
    # BB facing a pot raise
    g2 = Game(make_deal("insp", i), 1, i)
    g2.apply(Action.RAISE_POT)
    obs2 = g2.observation(0)
    a2 = brain.choose_action(obs2, memory)
    memory["hand"] = None
    # BB facing limp
    g3 = Game(make_deal("insp", i), 1, i)
    g3.apply(Action.CHECK_CALL)
    obs3 = g3.observation(0)
    a3 = brain.choose_action(obs3, memory)
    memory["hand"] = None
    rows.append((p, a, a2, a3))
rows = np.array(rows)
names = ["F", "C", "H", "P"]
for lo in np.arange(0, 1, 0.1):
    sel = rows[(rows[:, 0] >= lo) & (rows[:, 0] < lo + 0.1)]
    def dist(col):
        c = np.bincount(sel[:, col].astype(int), minlength=4) / len(sel)
        return " ".join(f"{n}{x:4.0%}" for n, x in zip(names, c))
    print(f"pct {lo:.1f}-{lo+0.1:.1f}  SB open: {dist(1)} | BB vs pot-raise: {dist(2)} | BB vs limp: {dist(3)}")
