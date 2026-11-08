"""Strategy switcher: alternates every PERIOD hands between an honest nit (bets only
strong hands, folds a lot) and a hyper-aggressive bluffer (bets/raises most hands, calls
down light). Built to punish opponents that learn slowly or never forget."""
import os
import numpy as np

from _common import Action, Street, can_raise, equity

PERIOD = int(os.environ.get("SWITCH_PERIOD", "150"))
START = int(os.environ.get("SWITCH_START", "0"))   # 0: begin as the nit, 1: begin as the maniac


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(121)


def _nit(obs, eq, rng):
    tc = obs.to_call
    price = tc / (obs.pot + tc) if tc else 0.0
    if obs.street == Street.PREFLOP:
        if eq > 0.58 and can_raise(obs):
            return Action.RAISE_POT
        if eq > 0.52 or tc == 0:
            return Action.CHECK_CALL
        return Action.FOLD
    if tc == 0:
        return Action.RAISE_POT if (eq > 0.75 and can_raise(obs)) else Action.CHECK_CALL
    if eq > 0.85 and can_raise(obs):
        return Action.RAISE_POT
    return Action.CHECK_CALL if eq > price + 0.12 else Action.FOLD


def _maniac(obs, eq, rng):
    tc = obs.to_call
    price = tc / (obs.pot + tc) if tc else 0.0
    if can_raise(obs) and (eq > 0.45 or rng.random() < 0.55):
        return Action.RAISE_POT if rng.random() < 0.6 else Action.RAISE_HALF
    if tc == 0 or eq > price - 0.12:
        return Action.CHECK_CALL
    return Action.FOLD


def choose_action(obs, memory):
    rng = memory["rng"]
    eq = equity(obs.hole_cards, obs.board, 120, rng)
    phase = (obs.hand_index // PERIOD + START) % 2
    return (_nit if phase == 0 else _maniac)(obs, eq, rng)


def on_hand_end(result, memory):
    pass
