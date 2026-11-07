"""Game settings. Check every default here against the official guide."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Rules:
    small_blind: int = 1
    big_blind: int = 2
    ante: int = 0
    starting_stack: int = 200          # each player starts every hand with this (stacks reset per hand)
    hole_cards: int = 5
    flush_beats_full_house: bool = True  # standard short deck: fewer flushes than full houses
    trips_beat_straight: bool = True     # Triton/GG short deck; set False if the guide says a straight wins

    def category_order(self) -> list[str]:
        """Hand categories from weakest to strongest under these rules."""
        order = ["high_card", "pair", "two_pair"]
        order += ["straight", "trips"] if self.trips_beat_straight else ["trips", "straight"]
        order += ["full_house", "flush"] if self.flush_beats_full_house else ["flush", "full_house"]
        order += ["quads", "straight_flush"]
        return order

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Rules":
        if not d:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def table_tag(self) -> str:
        return f"fbfh{int(self.flush_beats_full_house)}_tbs{int(self.trips_beat_straight)}"

    def describe(self) -> str:
        ranks = " < ".join(n.replace("_", " ") for n in self.category_order())
        return (f"blinds {self.small_blind}/{self.big_blind}, ante {self.ante}, "
                f"stack {self.starting_stack} ({self.starting_stack / self.big_blind:g} BB) reset every hand, "
                f"{self.hole_cards} hole cards, pot limit\n"
                f"hand ranks: {ranks}")
