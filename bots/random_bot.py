"""Practice bot: picks a random legal action and a random raise size."""
import random


class Bot:
    def __init__(self):
        self.rng = random.Random()

    def act(self, state):
        action = self.rng.choice(state["legal"])
        if action == "raise":
            return {"action": "raise",
                    "amount": self.rng.randint(state["min_raise_to"], state["max_raise_to"])}
        return {"action": action}
