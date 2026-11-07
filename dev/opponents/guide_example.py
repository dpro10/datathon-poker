"""The guide's section 4.1 example: call cheap bets before the river, raise pot with a flush+ on the river."""
from _common import Action, Street
from evaluator import FLUSH, omaha_score


def new_half(config, memory):
    pass


def choose_action(obs, memory):
    if obs.street != Street.RIVER:
        if obs.to_call <= obs.big_blind:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL
    score = omaha_score(obs.hole_cards, obs.board)
    if score >= (FLUSH << 20) and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT
    return Action.CHECK_CALL


def on_hand_end(result, memory):
    pass
