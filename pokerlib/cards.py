"""Cards for the 36-card short deck (6 through A).

A card is an int 0..35: rank_index * 4 + suit_index.
rank_index 0..8 is 6,7,8,9,T,J,Q,K,A and suit_index 0..3 is c,d,h,s.
Bots receive cards as strings like "As" or "Td"; parse_cards() converts them.
"""
from __future__ import annotations

from typing import Iterable

RANKS = "6789TJQKA"
SUITS = "cdhs"
NUM_CARDS = len(RANKS) * len(SUITS)  # 36


def rank_of(card: int) -> int:
    return card // 4


def suit_of(card: int) -> int:
    return card % 4


def card_str(card: int) -> str:
    return RANKS[card // 4] + SUITS[card % 4]


def cards_str(cards: Iterable[int]) -> list[str]:
    return [card_str(int(c)) for c in cards]


def parse_card(text: str) -> int:
    t = text.strip()
    if t[:2] == "10":
        t = "T" + t[2:]
    if len(t) != 2:
        raise ValueError(f"bad card {text!r}")
    r = RANKS.find(t[0].upper())
    s = SUITS.find(t[1].lower())
    if r < 0 or s < 0:
        raise ValueError(f"bad card {text!r} (short deck ranks are {RANKS}, suits {SUITS})")
    return r * 4 + s


def parse_cards(cards) -> list[int]:
    """Accepts "As Kd 7h", "AsKd7h", ["As", "Kd"], or a list of ints."""
    if cards is None:
        return []
    if isinstance(cards, str):
        parts = cards.replace(",", " ").split()
        if len(parts) == 1 and len(parts[0]) > 3:
            t = parts[0]
            parts = [t[i:i + 2] for i in range(0, len(t), 2)]
        return [parse_card(p) for p in parts]
    return [parse_card(c) if isinstance(c, str) else int(c) for c in cards]
