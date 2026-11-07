"""Configurable equity-vs-random threshold bot: the most common student design.

make(**params) returns (new_half, choose_action, on_hand_end) bound to its own
parameters, so several variants can live in one process.
  bet    open-bet when checked to at/above this equity (per street)
  pot    use RAISE_POT at/above, else RAISE_HALF
  raise  raise a bet at/above
  margin call when equity >= pot odds + margin
  bluff  chance to bet anyway when checked to
"""
import numpy as np

from _common import Action, can_raise, equity

DEFAULT = {
    "samples": 150,
    "bet": [0.55, 0.62, 0.66, 0.70],
    "pot": [0.60, 0.75, 0.80, 0.85],
    "raise": [0.60, 0.75, 0.80, 0.85],
    "margin": [0.00, 0.03, 0.03, 0.03],
    "bluff": 0.0,
    "seed": 7,
}


def make(**overrides):
    P = dict(DEFAULT, **overrides)

    def new_half(config, memory):
        memory["rng"] = np.random.default_rng(P["seed"])

    def choose_action(obs, memory):
        rng = memory["rng"]
        s = int(obs.street)
        eq = equity(obs.hole_cards, obs.board, P["samples"], rng)
        if obs.to_call == 0:
            if can_raise(obs) and (eq >= P["bet"][s] or rng.random() < P["bluff"]):
                return Action.RAISE_POT if eq >= P["pot"][s] else Action.RAISE_HALF
            return Action.CHECK_CALL
        if can_raise(obs) and eq >= P["raise"][s]:
            return Action.RAISE_POT if eq >= P["pot"][s] else Action.RAISE_HALF
        price = obs.to_call / (obs.pot + obs.to_call)
        if eq >= price + P["margin"][s]:
            return Action.CHECK_CALL
        return Action.FOLD

    def on_hand_end(result, memory):
        pass

    return new_half, choose_action, on_hand_end


new_half, choose_action, on_hand_end = make()
