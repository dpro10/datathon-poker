"""Baseline bot: estimates its chance to win, compares it with the price of calling,
and adjusts for how aggressive the opponent has been. Copy this to bots/<yourname>.py
and improve it. Every number in CAPS is a knob to tune with `python3 -m arena tournament`.

Reference points (equity vs one random hand): preflop the median hand is 0.50, the top 10%
is about 0.59 and the top 5% about 0.61. On the flop the median is 0.45 and the top 10% is 0.80.
"""
import numpy as np

from pokerlib import Rules, equity_vs_random, get_evaluator


class Bot:
    SAMPLES = {"preflop": 300, "flop": 500, "turn": 500, "river": 400}
    BET = {"preflop": 0.56, "flop": 0.62, "turn": 0.68, "river": 0.72}     # bet or open-raise at or above
    RAISE = {"preflop": 0.60, "flop": 0.75, "turn": 0.80, "river": 0.85}   # raise a bet at or above
    POT_SIZE_AT = 0.80      # use the biggest legal size at or above this equity
    CALL_MARGIN = 0.02      # call when equity beats the price by this much
    TIGHT_DISCOUNT = 0.10   # when a passive opponent bets, they usually have it

    def __init__(self):
        self.rng = np.random.default_rng()
        self.ev = get_evaluator(Rules())
        self.opp_decisions = 0
        self.opp_raises = 0

    def new_match(self, info):
        self.ev = get_evaluator(Rules.from_dict(info.get("rules")))

    def act(self, s):
        street = s["street"]
        eq = equity_vs_random(s["hole"], s["board"], n=self.SAMPLES[street], rng=self.rng, evaluator=self.ev)
        can_raise = "raise" in s["legal"]

        if s["to_call"] == 0:
            if can_raise and eq >= self.BET[street]:
                return {"action": "raise", "amount": self._size(s, eq)}
            return {"action": "check"}

        # Facing a bet. Equity against a random hand overstates our chances when the bettor is selective.
        aggression = (self.opp_raises + 1) / (self.opp_decisions + 3)
        eq -= self.TIGHT_DISCOUNT * max(0.0, 1 - aggression / 0.4)
        if can_raise and eq >= self.RAISE[street]:
            return {"action": "raise", "amount": self._size(s, eq)}
        price = s["to_call"] / (s["pot"] + s["to_call"])
        if eq >= price + self.CALL_MARGIN:
            return {"action": "call"}
        return {"action": "fold"}

    def _size(self, s, eq):
        lo, hi = s["min_raise_to"], s["max_raise_to"]
        if eq >= self.POT_SIZE_AT:
            return hi
        return int(lo + 0.66 * (hi - lo))

    def hand_over(self, result):
        for h in result["history"]:
            if h["actor"] == "opp" and h["action"] in ("fold", "check", "call", "raise"):
                self.opp_decisions += 1
                self.opp_raises += h["action"] == "raise"
