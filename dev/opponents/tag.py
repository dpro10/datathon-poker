"""Tight-aggressive heuristic bot with position awareness and c-bets."""
import numpy as np

from _common import Action, Street, can_raise, equity


def new_half(config, memory):
    memory["rng"] = np.random.default_rng(21)


def choose_action(obs, memory):
    rng = memory["rng"]
    eq = equity(obs.hole_cards, obs.board, 200, rng)
    price = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    if obs.street == Street.PREFLOP:
        if obs.to_call <= 1:  # SB first action or BB option
            if eq > 0.55 and can_raise(obs):
                return Action.RAISE_POT
            if eq > 0.47 or obs.to_call == 0:
                return Action.CHECK_CALL
            return Action.FOLD
        if eq > 0.60 and can_raise(obs):
            return Action.RAISE_POT
        if eq > 0.50 or (eq > 0.46 and obs.to_call <= 4):
            return Action.CHECK_CALL
        return Action.FOLD
    i_raised_pre = any(a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT) and a.street == Street.PREFLOP
                       for a in obs.action_history)
    if obs.to_call == 0:
        if can_raise(obs):
            if eq > 0.72:
                return Action.RAISE_POT
            if eq > 0.58 or (i_raised_pre and obs.street == Street.FLOP and rng.random() < 0.6):
                return Action.RAISE_HALF
        return Action.CHECK_CALL
    if eq > 0.82 and can_raise(obs):
        return Action.RAISE_POT
    if eq > price + 0.06:
        return Action.CHECK_CALL
    return Action.FOLD


def on_hand_end(result, memory):
    pass
