"""Sticky raiser: calls almost everything, raises pot with strong hands, never folds a pair+."""
import numpy as np

from _common import Action, Street, can_raise, equity


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(91)


def choose_action(obs, memory):
    eq = equity(obs.hole_cards, obs.board, 150, memory["rng"])
    if can_raise(obs) and eq > (0.58 if obs.street == Street.PREFLOP else 0.72):
        return Action.RAISE_POT
    if obs.to_call == 0 or eq > 0.22:
        return Action.CHECK_CALL
    return Action.FOLD


def on_hand_end(result, memory):
    pass
