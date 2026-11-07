"""Practice bot: never folds, never raises. Punish it by betting your good hands and never bluffing."""


class Bot:
    def act(self, state):
        return {"action": "check" if "check" in state["legal"] else "call"}
