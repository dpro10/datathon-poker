"""Bluffer: pot-bets whenever checked to, otherwise plays pot odds."""
import numpy as np

from _common import Action, can_raise, equity


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(51)


def choose_action(obs, memory):
    eq = equity(obs.hole_cards, obs.board, 150, memory["rng"])
    if obs.to_call == 0:
        return Action.RAISE_POT if can_raise(obs) else Action.CHECK_CALL
    if eq > 0.80 and can_raise(obs):
        return Action.RAISE_POT
    price = obs.to_call / (obs.pot + obs.to_call)
    return Action.CHECK_CALL if eq > price else Action.FOLD


def on_hand_end(result, memory):
    pass
