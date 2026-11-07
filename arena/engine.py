"""One hand of heads-up short-deck 5-card Pot-Limit Omaha. Integer chips, no I/O.

Seat 0 gets deck[0:5], seat 1 gets deck[5:10], the board is deck[10:15].
The button posts the small blind, acts first preflop and last on later streets.
Raise amounts are "raise to" totals for the current street.
"""
from __future__ import annotations

from pokerlib import Evaluator, Rules, cards_str

STREETS = ("preflop", "flop", "turn", "river")
_BOARD_SIZE = {0: 0, 1: 3, 2: 4, 3: 5}


class IllegalAction(Exception):
    pass


class Hand:
    def __init__(self, rules: Rules, evaluator: Evaluator, deck: list[int], button: int, hand_id: int = 0):
        h = rules.hole_cards
        self.rules = rules
        self.ev = evaluator
        self.hand_id = hand_id
        self.button = button
        self.hole = [list(deck[0:h]), list(deck[h:2 * h])]
        self.runout = list(deck[2 * h:2 * h + 5])
        self.board: list[int] = []
        self.street = 0
        self.stacks = [rules.starting_stack, rules.starting_stack]
        self.bets = [0, 0]        # this street
        self.contrib = [0, 0]     # whole hand
        self.acted = [False, False]
        self.last_raise = rules.big_blind
        self.history: list[dict] = []
        self.finished = False
        self.folded: int | None = None
        self.winner: int | None = None   # None with showdown=True means a split pot
        self.showdown = False
        self.values: list[int] | None = None
        self.net = [0, 0]

        if rules.ante:
            for s in (0, 1):
                a = min(rules.ante, self.stacks[s])
                self.stacks[s] -= a
                self.contrib[s] += a
                self._log(s, "ante", a)
        sb, bb = button, 1 - button
        self._put(sb, rules.small_blind)
        self._log(sb, "small_blind", self.bets[sb])
        self._put(bb, rules.big_blind)
        self._log(bb, "big_blind", self.bets[bb])
        self.to_act = sb
        self._advance()

    # ----- state -----
    @property
    def pot(self) -> int:
        return self.contrib[0] + self.contrib[1]

    def legal(self, s: int) -> dict:
        o = 1 - s
        to_call = max(0, self.bets[o] - self.bets[s])
        out = {"to_call": min(to_call, self.stacks[s])}
        actions = ["fold", "call"] if to_call > 0 else ["check"]
        if self.stacks[s] > to_call and self.stacks[o] > 0:
            effective_all_in = min(self.bets[s] + self.stacks[s], self.bets[o] + self.stacks[o])
            pot_limit = self.bets[o] + self.pot + to_call  # call first, then raise the size of the pot
            max_to = min(pot_limit, effective_all_in)
            min_to = min(self.bets[o] + max(self.last_raise, self.rules.big_blind), max_to)
            if max_to > self.bets[o]:
                actions.append("raise")
                out["min_raise_to"] = min_to
                out["max_raise_to"] = max_to
        out["actions"] = actions
        return out

    def view(self, s: int) -> dict:
        """What seat s is allowed to see when it is their turn."""
        lg = self.legal(s)
        d = {
            "hand_id": self.hand_id,
            "street": STREETS[self.street],
            "hole": cards_str(self.hole[s]),
            "board": cards_str(self.board),
            "button": self.button == s,
            "pot": self.pot,
            "my_stack": self.stacks[s],
            "opp_stack": self.stacks[1 - s],
            "my_bet": self.bets[s],
            "opp_bet": self.bets[1 - s],
            "to_call": lg["to_call"],
            "legal": lg["actions"],
            "history": self._history_for(s),
        }
        if "raise" in lg["actions"]:
            d["min_raise_to"] = lg["min_raise_to"]
            d["max_raise_to"] = lg["max_raise_to"]
        return d

    def result(self, s: int) -> dict:
        """Sent to seat s when the hand is over. Opponent cards only if there was a showdown."""
        o = 1 - s
        if self.winner is None:
            winner = "split"
        else:
            winner = "me" if self.winner == s else "opp"
        d = {
            "hand_id": self.hand_id,
            "net": self.net[s],
            "winner": winner,
            "showdown": self.showdown,
            "hole": cards_str(self.hole[s]),
            "opp_hole": cards_str(self.hole[o]) if self.showdown else None,
            "board": cards_str(self.board),
            "button": self.button == s,
            "history": self._history_for(s),
        }
        if self.showdown and self.values:
            d["my_hand"] = self.ev.category(self.values[s])
            d["opp_hand"] = self.ev.category(self.values[o])
        return d

    def record(self, name0: str, name1: str) -> dict:
        """Full hand history for log files."""
        names = (name0, name1)
        return {
            "hand_id": self.hand_id,
            "button": names[self.button],
            "hole": {names[0]: cards_str(self.hole[0]), names[1]: cards_str(self.hole[1])},
            "board": cards_str(self.board),
            "actions": [dict(e, seat=names[e["seat"]]) for e in self.history],
            "showdown": self.showdown,
            "net": {names[0]: self.net[0], names[1]: self.net[1]},
        }

    # ----- actions -----
    def act(self, s: int, action, amount=None) -> None:
        if self.finished:
            raise IllegalAction("hand is over")
        if s != self.to_act:
            raise IllegalAction("not your turn")
        lg = self.legal(s)
        if action not in lg["actions"]:
            raise IllegalAction(f"{action!r} is not legal here (legal: {', '.join(lg['actions'])})")
        o = 1 - s
        if action == "fold":
            self._log(s, "fold", 0)
            self.folded = s
            self.winner = o
            self.net = [0, 0]
            self.net[s] = -self.contrib[s]
            self.net[o] = self.contrib[s]
            self.finished = True
            return
        if action == "check":
            self._log(s, "check", 0)
        elif action == "call":
            self._put(s, lg["to_call"])
            self._log(s, "call", lg["to_call"])
        else:
            if amount is None:
                raise IllegalAction("raise needs an amount (the total you are raising to on this street)")
            try:
                amount = int(round(float(amount)))
            except (TypeError, ValueError):
                raise IllegalAction(f"bad raise amount {amount!r}")
            lo, hi = lg["min_raise_to"], lg["max_raise_to"]
            if not lo <= amount <= hi:
                raise IllegalAction(f"raise to {amount} is outside [{lo}, {hi}]")
            previous_high = self.bets[o]
            self._put(s, amount - self.bets[s])
            if amount - previous_high >= self.last_raise:
                self.last_raise = amount - previous_high
            self.acted = [False, False]
            self._log(s, "raise", amount)
        self.acted[s] = True
        self.to_act = o
        self._advance()

    # ----- internals -----
    def _put(self, s: int, amount: int) -> None:
        amount = min(amount, self.stacks[s])
        self.stacks[s] -= amount
        self.bets[s] += amount
        self.contrib[s] += amount

    def _log(self, s: int, action: str, amount: int) -> None:
        self.history.append({"street": STREETS[self.street], "seat": s, "action": action, "amount": amount})

    def _history_for(self, s: int) -> list[dict]:
        return [{"street": e["street"], "actor": "me" if e["seat"] == s else "opp",
                 "action": e["action"], "amount": e["amount"]} for e in self.history]

    def _round_over(self) -> bool:
        high = max(self.bets)
        for p in (0, 1):
            if self.stacks[p] == 0:
                continue
            if self.bets[p] < high:
                return False
            if not self.acted[p] and self.stacks[1 - p] > 0:
                return False
        return True

    def _advance(self) -> None:
        while not self.finished and self._round_over():
            if self.street == 3 or min(self.stacks) == 0:
                self.board = list(self.runout)
                self._showdown()
                return
            self.street += 1
            self.board = self.runout[:_BOARD_SIZE[self.street]]
            self.bets = [0, 0]
            self.acted = [False, False]
            self.last_raise = self.rules.big_blind
            self.to_act = 1 - self.button

    def _showdown(self) -> None:
        self.showdown = True
        self.values = [self.ev.best_omaha(self.hole[p], self.board) for p in (0, 1)]
        matched = min(self.contrib)  # anything above this was never called and goes back
        if self.values[0] == self.values[1]:
            self.winner = None
            self.net = [0, 0]
        else:
            w = 0 if self.values[0] > self.values[1] else 1
            self.winner = w
            self.net = [0, 0]
            self.net[w] = matched
            self.net[1 - w] = -matched
        self.finished = True
