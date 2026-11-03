"""Search the possible cards in the local scryfall data (<root>/scryfall/exp_<code>/info.json).

Offline: no internet request, only the info.json files written by download.py.

Criteria (a criterion left empty or '*' doesn't filter anything):
- exp_code: expansion code (ex: 'dsk')
- rarity: common, uncommon, rare, mythic, special (or its first letter: c, u, r, m, s)
- mana_cost: '{1}{W}{W}' or '1WW' (symbols in any order). Hybrid / Phyrexian symbols need
  braces: '{W/U}', '{G/P}'. 'none' = cards without mana cost (ex: lands). A double-faced card
  matches if one of its faces matches.

Run from anywhere:
    python3 src/search.py                                  (asks every criterion)
    python3 src/search.py dsk --rarity rare --mana-cost 1WW
"""
import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import local_paths

# Project root (<root>), independent of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
SCRYFALL_DIR = local_paths.SCRYFALL_DIR  # <root>/scryfall unless paths.json says otherwise

RARITIES = ["common", "uncommon", "rare", "mythic", "special", "bonus"]


@dataclass
class SearchCriteria:
    """What we know about the card. A criterion left to None doesn't filter anything."""
    exp_code: Optional[str] = None      # Expansion code, ex: 'dsk' (case insensitive)
    rarity: Optional[str] = None        # 'rare' or 'r' (case insensitive)
    mana_cost: Optional[str] = None     # '{1}{W}{W}' or '1WW', 'none' = no mana cost


# Cache of <root>/scryfall/exp_<code>/info.json, indexed by expansion code.
_cards_by_exp: Dict[str, List[dict]] = {}


def list_exp_codes() -> List[str]:
    """Every expansion code having a scryfall/exp_<code>/info.json."""
    return sorted(p.parent.name[len("exp_"):] for p in SCRYFALL_DIR.glob("exp_*/info.json"))


def load_cards(exp_code: str) -> List[dict]:
    """Cards of scryfall/exp_<exp_code>/info.json (cached). Empty list if the file doesn't exist."""
    exp_code = exp_code.lower()
    if exp_code not in _cards_by_exp:
        info_path = SCRYFALL_DIR / f"exp_{exp_code}" / "info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as fp:
                _cards_by_exp[exp_code] = json.load(fp)
        else:
            _cards_by_exp[exp_code] = []
    return _cards_by_exp[exp_code]


def _rarity(text: str) -> str:
    """'R' -> 'rare', 'Mythic' -> 'mythic'. Unknown values are kept (lower case)."""
    text = text.strip().lower()
    return next((rarity for rarity in RARITIES if rarity == text or rarity[0] == text), text)


def _mana_symbols(text: str) -> Counter:
    """
    Mana symbols of a cost, in any order: '{1}{W}{W}' and '1WW' -> {'1': 1, 'W': 2}.
    Without braces, a number is one symbol ('10U' -> 10, U) and each letter is one symbol.
    """
    text = text.strip().upper()
    if "{" in text:
        return Counter(re.findall(r"\{([^}]*)\}", text))
    return Counter(re.findall(r"\d+|[A-Z]", text))


def _face_costs(card: dict) -> List[str]:
    """Mana cost of each face ('{U} // {4}{U}' -> ['{U}', '{4}{U}'])."""
    faces = card.get("card_faces") or []
    costs = [face.get("mana_cost", "") for face in faces if "mana_cost" in face]
    return costs or [part.strip() for part in card.get("mana_cost", "").split("//")]


def _matches_mana_cost(card: dict, wanted: str) -> bool:
    if wanted.strip().lower() == "none":
        return any(not cost for cost in _face_costs(card))
    symbols = _mana_symbols(wanted)
    return any(_mana_symbols(cost) == symbols for cost in _face_costs(card))


def _collector_key(card: dict):
    """Sort by collector number as a number: '2' < '10' < '10a' < '★1'."""
    number = card.get("collector_number", "")
    match = re.match(r"(\d+)(.*)", number)
    return (0, int(match.group(1)), match.group(2)) if match else (1, 0, number)


def search(criteria: SearchCriteria) -> List[dict]:
    """
    :return: the cards (scryfall json objects) matching every criterion, sorted by expansion
             then collector number.
    """
    exp_codes = [criteria.exp_code.lower()] if criteria.exp_code else list_exp_codes()
    rarity = _rarity(criteria.rarity) if criteria.rarity else None
    cards = []
    for exp_code in exp_codes:
        for card in load_cards(exp_code):
            if rarity and card.get("rarity") != rarity:
                continue
            if criteria.mana_cost and not _matches_mana_cost(card, criteria.mana_cost):
                continue
            cards.append(card)
    return sorted(cards, key=lambda card: (card.get("set", ""), _collector_key(card)))


def _field(card: dict, name: str) -> str:
    """Field of a card; for a double-faced card without it, the faces' values joined by ' // '."""
    if card.get(name):
        return str(card[name])
    faces = card.get("card_faces") or []
    return " // ".join(str(face.get(name, "")) for face in faces if face.get(name))


def print_cards(cards: List[dict], criteria: SearchCriteria):
    """Print the cards as a table: collector number, name, rarity, mana cost, type."""
    if not cards:
        print(f"No card found for {criteria}.")
        if criteria.exp_code and criteria.exp_code.lower() not in list_exp_codes():
            print(f"No scryfall/exp_{criteria.exp_code.lower()}/info.json: download the expansion first "
                  f"(python3 src/download.py). Known codes: {', '.join(list_exp_codes())}")
        return

    expansions = {card.get("set", ""): card.get("set_name", "") for card in cards}
    title = ", ".join(f"{name} ({code})" for code, name in expansions.items())
    print(f"{len(cards)} possible card(s) - {title}")

    headers = ["Set", "#", "Name", "Rarity", "Mana cost", "Type"]
    rows = [[card.get("set", ""), card.get("collector_number", ""), card.get("name", ""),
             card.get("rarity", ""), _field(card, "mana_cost"), _field(card, "type_line")] for card in cards]
    if len(expansions) == 1:  # The set is already in the title
        headers, rows = headers[1:], [row[1:] for row in rows]
    widths = [max(len(str(value)) for value in column) for column in zip(headers, *rows)]
    for row in [headers, ["-" * width for width in widths]] + rows:
        print("  ".join(str(value).ljust(width) for value, width in zip(row, widths)).rstrip())


def _criterion(value: Optional[str]) -> Optional[str]:
    """Empty answer or '*' = any value (None)."""
    value = (value or "").strip()
    return None if value in ("", "*") else value


def main():
    parser = argparse.ArgumentParser(description="List the possible cards from the local scryfall data.")
    parser.add_argument("exp_code", nargs="?", help="expansion code, ex: dsk ('*' = every expansion)")
    parser.add_argument("--rarity", help="common, uncommon, rare, mythic, special (or c, u, r, m, s)")
    parser.add_argument("--mana-cost", help="ex: {1}{W}{W} or 1WW, 'none' = no mana cost")
    args = parser.parse_args()

    if args.exp_code is None and args.rarity is None and args.mana_cost is None:
        # No argument: ask every criterion. Empty answer = '*'.
        args.exp_code = input("Expansion code (empty = *) : ")
        args.rarity = input("Rarity: common, uncommon, rare, mythic, special (empty = *) : ")
        args.mana_cost = input("Mana cost, ex: {1}{W}{W} or 1WW (empty = *) : ")
    criteria = SearchCriteria(exp_code=_criterion(args.exp_code), rarity=_criterion(args.rarity),
                              mana_cost=_criterion(args.mana_cost))
    print_cards(search(criteria), criteria)


if __name__ == "__main__":
    main()
