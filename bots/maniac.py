"""Practice bot: raises the maximum every time it can. Beat it by calling down with decent hands."""


class Bot:
    def act(self, state):
        if "raise" in state["legal"]:
            return {"action": "raise", "amount": state["max_raise_to"]}
        return {"action": "check" if "check" in state["legal"] else "call"}
