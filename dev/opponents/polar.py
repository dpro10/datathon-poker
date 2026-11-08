"""Polarized bluffer: when it can bet, it bets pot with its best ~25% and worst ~30% of
hands and checks the middle; it raises polar too, and defends close to minimum-defense
frequency. Its bets look like value bets until showdowns reveal the bluffs."""
import numpy as np

from _common import Action, Street, can_raise, equity


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(131)


def choose_action(obs, memory):
    rng = memory["rng"]
    eq = equity(obs.hole_cards, obs.board, 160, rng)
    tc = obs.to_call
    if obs.street == Street.PREFLOP:
        if can_raise(obs) and (eq > 0.57 or eq < 0.40):
            return Action.RAISE_POT
        return Action.CHECK_CALL if (eq > 0.36 or tc == 0) else Action.FOLD
    # strength thresholds by street: equity vs a random hand spreads out as cards come
    top = {1: 0.68, 2: 0.72, 3: 0.75}[int(obs.street)]
    bottom = {1: 0.28, 2: 0.25, 3: 0.22}[int(obs.street)]
    if tc == 0:
        if can_raise(obs) and (eq > top or eq < bottom):
            return Action.RAISE_POT
        return Action.CHECK_CALL
    if can_raise(obs) and (eq > 0.86 or (eq < 0.15 and rng.random() < 0.35)):
        return Action.RAISE_POT
    bet_frac = tc / max(obs.pot - tc, 1)
    defend = 0.32 + 0.12 * min(bet_frac, 1.0)
    return Action.CHECK_CALL if eq > defend else Action.FOLD


def on_hand_end(result, memory):
    pass
