"""Balanced-style bot: polarized betting (value + bluffs), defends near minimum-defense
frequency, 3-bets with a value + bluff mix. Harder to exploit than threshold bots."""
import numpy as np

from _common import Action, Street, can_raise, equity


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(81)


def choose_action(obs, memory):
    rng = memory["rng"]
    eq = equity(obs.hole_cards, obs.board, 200, rng)
    tc = obs.to_call
    if obs.street == Street.PREFLOP:
        if obs.raises_this_street == 0 and tc > 0:           # SB open
            if eq > 0.53 and can_raise(obs):
                return Action.RAISE_POT
            return Action.CHECK_CALL if eq > 0.38 else Action.FOLD
        if tc == 0:                                          # BB vs limp
            if can_raise(obs) and (eq > 0.56 or (eq < 0.40 and rng.random() < 0.25)):
                return Action.RAISE_POT
            return Action.CHECK_CALL
        if can_raise(obs) and (eq > 0.60 or (0.40 < eq < 0.44 and rng.random() < 0.4)):
            return Action.RAISE_POT
        return Action.CHECK_CALL if eq > 0.42 else Action.FOLD
    if tc == 0:
        if can_raise(obs):
            if eq > 0.78:
                return Action.RAISE_POT
            if eq > 0.66 or (eq < 0.25 and rng.random() < 0.45):
                return Action.RAISE_HALF if rng.random() < 0.5 else Action.RAISE_POT
        return Action.CHECK_CALL
    bet_frac = tc / max(obs.pot - tc, 1)
    mdf_thresh = 0.30 + 0.15 * min(bet_frac, 1.0)             # call roughly the top 55-70% of hands
    if can_raise(obs) and (eq > 0.88 or (eq < 0.2 and rng.random() < 0.12)):
        return Action.RAISE_POT
    return Action.CHECK_CALL if eq > mdf_thresh else Action.FOLD


def on_hand_end(result, memory):
    pass
