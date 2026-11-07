"""Tight equity bot (nit-ish thresholds)."""
from eqbot import make

new_half, choose_action, on_hand_end = make(
    bet=[0.58, 0.70, 0.74, 0.78], pot=[0.62, 0.80, 0.84, 0.88],
    **{"raise": [0.62, 0.82, 0.85, 0.88]}, margin=[0.04, 0.08, 0.08, 0.08], seed=11)
