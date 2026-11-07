"""The unmodified starter bot: check when free, otherwise fold."""
from _common import Action


def new_half(config, memory):
    pass


def choose_action(obs, memory):
    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def on_hand_end(result, memory):
    pass
