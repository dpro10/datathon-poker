"""Betting state for one hand, replicating game.py's rules exactly (sizes, caps, street ends).

Seats here are 0 = me, 1 = opponent. Amounts are chips. Blinds, stacks and the raise
cap come from the match config the judge sends with /new-half (see set_rules), so a
change of settings before the tournament doesn't break the replay.
"""
from __future__ import annotations

FOLD, CALL, HALF, POT = 0, 1, 2, 3          # same ints as game.Action
SB, BB, STACK, CAP = 1, 2, 80, 4          # defaults; the live values are in RULES
PRE, FLOP, TURN, RIVER = 0, 1, 2, 3
RULES = {"sb": SB, "bb": BB, "stack": STACK, "cap": CAP}


def set_rules(small_blind=SB, big_blind=BB, starting_stack=STACK, raise_cap=CAP) -> None:
    """Use the judge's match settings (called from new_half)."""
    RULES.update(sb=int(small_blind), bb=int(big_blind), stack=int(starting_stack), cap=int(raise_cap))


class BState:
    __slots__ = ("street", "bets", "stacks", "contrib", "acted", "raises", "last_inc", "to_act",
                 "button", "over", "folded", "showdown", "last_raise", "bb", "cap")

    def __init__(self, me_button: bool):
        r = RULES
        self.bb, self.cap = r["bb"], r["cap"]
        self.street = PRE
        self.button = 0 if me_button else 1
        self.stacks = [r["stack"], r["stack"]]
        self.contrib = [0, 0]
        self.bets = [0, 0]
        self.acted = [False, False]
        self.raises = 0
        self.last_inc = self.bb
        self.over = False
        self.folded = -1          # seat that folded
        self.showdown = False
        self.last_raise = -1      # HALF or POT for the latest raise on this street, -1 if none
        b = self.button
        self._put(b, r["sb"])
        self._put(1 - b, self.bb)
        self.to_act = b

    def copy(self) -> "BState":
        n = BState.__new__(BState)
        n.street, n.button, n.raises, n.last_inc = self.street, self.button, self.raises, self.last_inc
        n.stacks, n.contrib, n.bets, n.acted = self.stacks[:], self.contrib[:], self.bets[:], self.acted[:]
        n.over, n.folded, n.showdown, n.to_act = self.over, self.folded, self.showdown, self.to_act
        n.last_raise = self.last_raise
        n.bb, n.cap = self.bb, self.cap
        return n

    @property
    def pot(self) -> int:
        return self.contrib[0] + self.contrib[1]

    def _put(self, seat, amount):
        self.stacks[seat] -= amount
        self.bets[seat] += amount
        self.contrib[seat] += amount

    def to_call(self, seat=None) -> int:
        s = self.to_act if seat is None else seat
        return max(0, self.bets[1 - s] - self.bets[s])

    def can_raise(self, seat=None) -> bool:
        s = self.to_act if seat is None else seat
        return self.raises < self.cap and self.stacks[s] > self.to_call(s) and self.stacks[1 - s] > 0

    def raise_to(self, action, seat=None) -> int:
        s = self.to_act if seat is None else seat
        my_bet, opp_bet = self.bets[s], self.bets[1 - s]
        tc = max(0, opp_bet - my_bet)
        pot = self.contrib[0] + self.contrib[1]
        all_in_to = my_bet + self.stacks[s]
        pot_to = opp_bet + pot + tc
        half_to = opp_bet + (pot + tc) // 2
        target = half_to if action == HALF else pot_to
        min_to = opp_bet + max(self.last_inc, self.bb)
        return min(max(target, min_to), pot_to, all_in_to)

    def legal(self):
        if self.over:
            return ()
        s = self.to_act
        acts = [FOLD] if self.to_call(s) > 0 else []
        acts.append(CALL)
        if self.can_raise(s):
            acts += [HALF, POT]
        return tuple(acts)

    def apply(self, action) -> int:
        """Play `action` for the seat to act; returns the chips it added."""
        s, o = self.to_act, 1 - self.to_act
        if action == FOLD:
            self.over, self.folded, self.to_act = True, s, -1
            return 0
        if action == CALL:
            amount = min(self.to_call(s), self.stacks[s])
        else:
            new_total = self.raise_to(action, s)
            amount = new_total - self.bets[s]
            self.last_inc = new_total - self.bets[o]
            self.raises += 1
            self.last_raise = action
            self.acted[o] = False
        self._put(s, amount)
        self.acted[s] = True
        self._advance()
        return amount

    def _advance(self):
        if not (self.acted[0] and self.acted[1] and self.bets[0] == self.bets[1]):
            self.to_act = 1 - self.to_act
            return
        if self.street == RIVER or 0 in self.stacks:
            self.street = RIVER if self.street == RIVER else self.street
            self.over, self.showdown, self.to_act = True, True, -1
            return
        self.street += 1
        self.bets = [0, 0]
        self.acted = [False, False]
        self.raises = 0
        self.last_inc = self.bb
        self.last_raise = -1
        self.to_act = 1 - self.button

    def street_ended_after(self, street_before: int) -> bool:
        return self.over or self.street != street_before
