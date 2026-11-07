"""Practice bot: plays only strong hands and never bluffs. Steal its blinds, fold when it raises."""
import numpy as np

from pokerlib import Rules, equity_vs_random, get_evaluator


class Bot:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.ev = get_evaluator(Rules())

    def new_match(self, info):
        self.ev = get_evaluator(Rules.from_dict(info.get("rules")))

    def act(self, s):
        eq = equity_vs_random(s["hole"], s["board"], n=300, rng=self.rng, evaluator=self.ev)
        preflop = s["street"] == "preflop"
        if "raise" in s["legal"] and eq >= (0.60 if preflop else 0.85):
            return {"action": "raise", "amount": s["max_raise_to"]}
        if s["to_call"] == 0:
            return {"action": "check"}
        price = s["to_call"] / (s["pot"] + s["to_call"])
        needed = max(0.53, price + 0.05) if preflop else price + 0.15
        return {"action": "call" if eq >= needed else "fold"}
