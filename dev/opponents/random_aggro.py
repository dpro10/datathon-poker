"""Random but aggressive: raises 60% of the time when it can, folds 15% when facing a bet."""
import numpy as np

from _common import Action


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(101)


def choose_action(obs, memory):
    r = memory["rng"].random()
    legal = obs.legal_actions
    if Action.RAISE_POT in legal and r < 0.6:
        return Action.RAISE_POT if r < 0.35 else Action.RAISE_HALF
    if Action.FOLD in legal and r > 0.85:
        return Action.FOLD
    return Action.CHECK_CALL


def on_hand_end(result, memory):
    pass
