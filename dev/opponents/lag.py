"""Loose-aggressive bot: plays many hands, bets and bluffs a lot, calls light."""
import numpy as np

from _common import Action, Street, can_raise, equity


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(31)


def choose_action(obs, memory):
    rng = memory["rng"]
    eq = equity(obs.hole_cards, obs.board, 150, rng)
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if obs.street == Street.PREFLOP:
        if eq > 0.50 and can_raise(obs) and obs.raises_this_street < 3:
            return Action.RAISE_POT if eq > 0.56 else Action.RAISE_HALF
        if eq > 0.40 or obs.to_call == 0:
            return Action.CHECK_CALL
        return Action.FOLD
    if obs.to_call == 0:
        if can_raise(obs) and (eq > 0.55 or rng.random() < 0.40):
            return Action.RAISE_POT if (eq > 0.70 or rng.random() < 0.5) else Action.RAISE_HALF
        return Action.CHECK_CALL
    if can_raise(obs) and (eq > 0.72 or (rng.random() < 0.10 and obs.raises_this_street < 2)):
        return Action.RAISE_POT
    if eq > price - 0.05:
        return Action.CHECK_CALL
    return Action.FOLD


def on_hand_end(result, memory):
    pass
