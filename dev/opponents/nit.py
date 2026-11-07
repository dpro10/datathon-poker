"""Nit: plays only strong hands, folds to pressure unless very strong."""
import numpy as np

from _common import Action, Street, can_raise, equity


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(41)


def choose_action(obs, memory):
    rng = memory["rng"]
    eq = equity(obs.hole_cards, obs.board, 200, rng)
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if obs.street == Street.PREFLOP:
        if eq > 0.60 and can_raise(obs):
            return Action.RAISE_POT
        if eq > 0.56 or obs.to_call == 0:
            return Action.CHECK_CALL
        return Action.FOLD
    if obs.to_call == 0:
        if eq > 0.75 and can_raise(obs):
            return Action.RAISE_POT
        return Action.CHECK_CALL
    if eq > 0.88 and can_raise(obs):
        return Action.RAISE_POT
    if eq > price + 0.15:
        return Action.CHECK_CALL
    return Action.FOLD


def on_hand_end(result, memory):
    pass
