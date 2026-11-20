import requests
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import local_paths


HEADERS = {"User-Agent": "mtg-corner-detector/1.0"}

# Project root (<root>), independent of the current working directory.
ROOT = local_paths.ROOT
# <root>/scryfall unless paths.json says otherwise (see local_paths.py).
SCRYFALL_DIR = local_paths.SCRYFALL_DIR
# The cards (info.json + images): <scryfall dir>/exp/exp_<code>/.
EXP_DIR = local_paths.EXP_DIR

# The card symbols ({T}, {W}, {W/U}...): <scryfall dir>/symbol/*.svg + symbology.json (see download_symbols).
SYMBOLOGY_URL = "https://api.scryfall.com/symbology"
SYMBOL_DIR_NAME = "symbol"

# The set symbols (download_set_symbols), all in <scryfall dir>/set/: set_<code>.svg as downloaded,
# set_<code>_<rarity>.svg the same symbol in the colors of a rarity (see colorize_set_symbol.py), and sets.json
# (the scryfall sets, with "symbol_file": the downloaded file, null = the set has no symbol).
SETS_URL = "https://api.scryfall.com/sets"
SET_SYMBOL_URL = "https://svgs.scryfall.io/sets/{code}.svg"
SET_DIR_NAME = "set"
SETS_FILE_NAME = "sets.json"
SET_SYMBOL_FILE_KEY = "symbol_file"

expension = {}

# Cache of <root>/scryfall/exp/exp_<exp_code>/info.json, indexed by card id.
_cards_by_id: Dict[str, Dict[str, dict]] = {}


def list_set():
    """Get information about the most recent MTG set"""
    sets_url = "https://api.scryfall.com/sets"
    response = requests.get(sets_url, headers=HEADERS)
    sets_data = response.json()

    # Find the most recent standard set
    # Sets are typically ordered by release date
    for set_data in sets_data["data"]:
        expension[set_data["code"]] = set_data
        print(f"> {set_data['code']} : {set_data['name']}")


def get_card_images(card: dict, image_type: str = "normal") -> List[Tuple[str, str, dict]]:
    """
    List every image of a card, handling double-faced cards.

    :return: a list of (illustration_id, image_url, face) where `face` is the json object
             holding the image (the card itself for a single-faced card, the face otherwise).
    """
    images = []
    # Some cards have multiple faces
    if "image_uris" in card:
        # Single-faced card
        images.append((card.get("illustration_id", card["id"]), card["image_uris"].get(image_type), card))
    elif "card_faces" in card:
        # Double-faced card
        for i, face in enumerate(card["card_faces"]):
            if "image_uris" in face:
                illustration_id = face.get("illustration_id", f"{card['id']}_{i}")
                images.append((illustration_id, face["image_uris"].get(image_type), face))
    return images


def download_expension(exp_code: str, image: bool):
    """
    Save the cards of a set in <root>/scryfall/exp/exp_<exp_code>/info.json (+ their missing images if `image`).
    An existing info.json is reused, not downloaded again: delete it to get the latest data.
    """
    exp_dir = EXP_DIR / f"exp_{exp_code}"
    info_path = exp_dir / "info.json"

    if info_path.exists():
        print(f"{info_path} already exists, reused.")
        with open(info_path, "r", encoding="utf-8") as fp:
            cards = json.load(fp)
    else:
        cards = _search_cards(exp_code)
        if not cards:
            return 0  # No info.json: the set is tried again next time
        os.makedirs(exp_dir, exist_ok=True)
        with open(info_path, "w", encoding="utf-8") as fp:
            json.dump(cards, fp, indent=4)

    if image:
        for card in cards:
            download_card_images(exp_code, card)
    return len(cards)


def _search_cards(exp_code: str) -> List[dict]:
    """Every card of a set from the scryfall search API (all pages). Empty list if the set has no card."""
    cards = []
    next_page_url = f"https://api.scryfall.com/cards/search?q=set:{exp_code}&unique=cards"
    page = 1
    while next_page_url:
        print(f"Processing page {page}...")
        response = requests.get(next_page_url, headers=HEADERS)

        # No card found for the expension code.
        if response.status_code == 404:
            print(f"No cards found for this expension.")
            break

        # Check for rate limiting or errors
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            time.sleep(1)  # Wait before retry
            continue

        data = response.json()
        cards.extend(data.get("data", []))
        next_page_url = data.get("next_page") if data.get("has_more") else None

        # Respect rate limits - pause between pages
        page += 1
        time.sleep(0.5)
    return cards


def save_image(image_url, directory, filename):
    """Download and save an image"""
    if not image_url:
        return False

    try:
        response = requests.get(image_url, headers=HEADERS)
        if response.status_code == 200:
            file_path = os.path.join(directory, f"{filename}.jpg")
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
            return True
        else:
            print(f"Failed to download {filename}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False


def _get_card(exp_code: str, card_id: str) -> dict:
    """Find a card by its scryfall id in <root>/scryfall/exp/exp_<exp_code>/info.json (cached)."""
    if exp_code not in _cards_by_id:
        with open(EXP_DIR / f"exp_{exp_code}" / "info.json", "r", encoding="utf-8") as fp:
            _cards_by_id[exp_code] = {card["id"]: card for card in json.load(fp)}
    if card_id not in _cards_by_id[exp_code]:
        raise KeyError(f"Card '{card_id}' not found in expension '{exp_code}'")
    return _cards_by_id[exp_code][card_id]


def download_card_images(exp_code: str, card: dict) -> List[str]:
    """
    Download every image of a card (all faces for a double-faced card) that is not already
    in <root>/scryfall/exp/exp_<exp_code>/image/<illustration_id>.jpg.

    :param exp_code: expension code (ex: 'mrd')
    :param card: scryfall json object of the card
    :return: the path of every image of the card available on disk (already present or just
             downloaded), in face order. A face whose download failed is not returned.
    """
    image_dir = EXP_DIR / f"exp_{exp_code}" / "image"
    os.makedirs(image_dir, exist_ok=True)

    paths = []
    for illustration_id, image_url, _ in get_card_images(card):
        path = image_dir / f"{illustration_id}.jpg"
        if not path.exists():
            downloaded = save_image(image_url, image_dir, f"{illustration_id}")
            # Be nice to the API - don't flood it with requests
            time.sleep(0.1)
            if not downloaded:
                continue
        paths.append(str(path))
    return paths


def download_if_not_exist(exp_code: str, card_id: str) -> List[str]:
    """download_card_images for a card found by its scryfall id in <root>/scryfall/exp/exp_<exp_code>/info.json."""
    return download_card_images(exp_code, _get_card(exp_code, card_id))


# ---------------------------------------------------------------------------
# Card symbols (https://scryfall.com/docs/api/card-symbols/all)
# ---------------------------------------------------------------------------

def symbol_dir(name: str = SYMBOL_DIR_NAME) -> Path:
    """
    The folder of the symbols: SCRYFALL_DIR/<name> (name: 'symbol', 'set' or 'set_png'). In WSL with
    paths.json, on the Linux disk only (ex: ~/mtg/scryfall/symbol): nothing is written in <root>/scryfall.
    """
    return SCRYFALL_DIR / name


def symbol_file_stem(symbol: str) -> str:
    """
    File name (without .svg) of a symbol, same as mana_cost_gen_dataset.py: '{W/U}' -> 'W-U', '{T}' -> 'T'.
    The other characters that Windows refuses in a file name become '_'.
    """
    return re.sub(r'[<>:"\\|?*]', "_", symbol.strip("{}").replace("/", "-"))


def _list_all(url: str) -> List[dict]:
    """The "data" of a scryfall list (all pages)."""
    items = []
    while url:
        response = requests.get(url, headers={**HEADERS, "Accept": "application/json"}, timeout=30)
        response.raise_for_status()
        page = response.json()
        items.extend(page.get("data", []))
        url = page.get("next_page") if page.get("has_more") else None
        if url:
            time.sleep(0.1)  # Respect rate limits
    return items


def list_symbols() -> List[dict]:
    """Every card symbol of scryfall (all pages)."""
    return _list_all(SYMBOLOGY_URL)


def _write_file(path: Path, content: bytes):
    """Write through a .part file: a download stopped halfway never leaves a broken file (it would be kept)."""
    part_path = path.with_name(path.name + ".part")
    part_path.write_bytes(content)
    os.replace(part_path, path)


def _download_svg(url: str) -> Optional[bytes]:
    """Content of an SVG file, None (with a message) if the download fails."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as error:
        print(f"Error downloading {url}: {error}")
        return None
    if response.status_code != 200 or b"<svg" not in response.content:
        print(f"Failed to download {url}: HTTP {response.status_code}")
        return None
    return response.content


def download_symbols(force: bool = False) -> Dict[str, int]:
    """
    Every scryfall card symbol as an SVG file ({W/U} -> W-U.svg) in symbol_dir(), with symbology.json:
    what scryfall says about each symbol (english, colors, mana_value, appears_in_mana_costs...).
    The files already there are kept (force: download them again).
    :return: number of symbols 'downloaded', 'kept' (already there), 'failed'.
    """
    directory = symbol_dir()
    symbols = list_symbols()
    with_svg = [symbol for symbol in symbols if symbol.get("svg_uri")]
    without_svg = [symbol["symbol"] for symbol in symbols if not symbol.get("svg_uri")]

    # Two symbols can't share a file (Windows ignores the case of the file names).
    by_file: Dict[str, List[str]] = {}
    for symbol in with_svg:
        by_file.setdefault(symbol_file_stem(symbol["symbol"]).lower(), []).append(symbol["symbol"])
    clashes = [names for names in by_file.values() if len(names) > 1]
    if clashes:
        raise ValueError(f"Several symbols would get the same file: {clashes}. Change symbol_file_stem.")

    directory.mkdir(parents=True, exist_ok=True)
    print(f"{len(with_svg)} symbols for: {directory}")

    counts = {"downloaded": 0, "kept": 0, "failed": 0}
    failed = []
    for symbol in with_svg:
        name = f"{symbol_file_stem(symbol['symbol'])}.svg"
        path = directory / name
        if not force and path.exists():
            counts["kept"] += 1
            continue
        content = _download_svg(symbol["svg_uri"])
        time.sleep(0.1)  # Be nice to the API - don't flood it with requests
        if content is None:
            failed.append(symbol["symbol"])
            continue
        _write_file(path, content)
        counts["downloaded"] += 1
        print(f"Downloaded: {symbol['symbol']} -> {name}")

    _write_file(directory / "symbology.json", json.dumps(symbols, indent=4, ensure_ascii=False).encode("utf-8"))

    counts["failed"] = len(failed)
    print(f"Download complete. {len(with_svg)} symbols: {counts['downloaded']} downloaded, "
          f"{counts['kept']} already there.")
    if without_svg:
        print(f"No SVG on scryfall, skipped: {' '.join(without_svg)}")
    if failed:
        print(f"Failed, run it again: {' '.join(failed)}")
    return counts


# ---------------------------------------------------------------------------
# Set symbols (https://scryfall.com/docs/api/sets)
# ---------------------------------------------------------------------------

def list_sets() -> List[dict]:
    """Every scryfall set (code, name, released_at, set_type...)."""
    return _list_all(SETS_URL)


def set_symbol_file_name(code: str) -> str:
    """File of the downloaded symbol of a set: 'set_<code>.svg'."""
    return f"set_{code}.svg"


def _get_set_svg(url: str) -> Tuple[Optional[bytes], str]:
    """
    (content, 'ok'), or (None, 'missing') when scryfall has no such file (HTTP 403 / 404: the set has
    no symbol of its own, it's normal), or (None, 'failed') with a message (network, server...).
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as error:
        print(f"Error downloading {url}: {error}")
        return None, "failed"
    if response.status_code in (403, 404):
        return None, "missing"
    if response.status_code != 200 or b"<svg" not in response.content:
        print(f"Failed to download {url}: HTTP {response.status_code}")
        return None, "failed"
    return response.content, "ok"


def _load_symbol_files(directory: Path) -> Dict[str, Optional[str]]:
    """What the previous runs found, from the sets.json of `directory`: code -> file (None = no symbol)."""
    path = directory / SETS_FILE_NAME
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fp:
        return {set_data["code"]: set_data[SET_SYMBOL_FILE_KEY] for set_data in json.load(fp)
                if SET_SYMBOL_FILE_KEY in set_data}


def download_set_symbols(codes: Optional[List[str]] = None, force: bool = False) -> Dict[str, int]:
    """
    The symbol of every scryfall set (or only `codes`): SET_SYMBOL_URL as set_<code>.svg, unchanged, in
    symbol_dir(SET_DIR_NAME), then the same symbol in the colors of each rarity in the same folder:
    set_<code>_<rarity>.svg, and its PNG in set_png/ (colorize_set_symbol.colorize_set_symbols).
    Many sets have no symbol of their own (tokens, promos...): it's normal. sets.json remembers them
    ("symbol_file": null) and they are not asked again. The files already there are kept.
    force: ask scryfall again for every set, and draw the rarity symbols again.
    :return: number of sets 'downloaded', 'kept' (already there), 'no_symbol', 'failed'.
    """
    directory = symbol_dir(SET_DIR_NAME)
    directory.mkdir(parents=True, exist_ok=True)
    sets = list_sets()
    by_code = {set_data["code"]: set_data for set_data in sets}
    if codes is None:
        wanted = list(by_code)
    else:
        wanted = []
        for code in (code.strip().lower() for code in codes):
            if code in by_code:
                wanted.append(code)
            else:
                print(f"'{code}' is not a scryfall set, skipped.")
    print(f"{len(wanted)} set symbols for: {directory}")

    symbol_files = _load_symbol_files(directory)
    counts = {"downloaded": 0, "kept": 0, "no_symbol": 0, "failed": 0}
    no_symbol, failed = [], []
    for code in wanted:
        name = set_symbol_file_name(code)
        path = directory / name
        if not force and path.exists():
            symbol_files[code] = name
            counts["kept"] += 1
            continue
        if not force and code in symbol_files and symbol_files[code] is None:
            no_symbol.append(code)  # Already asked: scryfall has no symbol for this set
            continue
        content, status = _get_set_svg(SET_SYMBOL_URL.format(code=code))
        time.sleep(0.1)  # Be nice to the API - don't flood it with requests
        if status == "missing":
            symbol_files[code] = None
            no_symbol.append(code)
            continue
        if content is None:
            failed.append(code)
            continue
        _write_file(path, content)
        symbol_files[code] = name
        counts["downloaded"] += 1
        print(f"Downloaded: {code} ({by_code[code]['name']}) -> {name}")

    for set_data in sets:
        if set_data["code"] in symbol_files:
            set_data[SET_SYMBOL_FILE_KEY] = symbol_files[set_data["code"]]
    _write_file(directory / SETS_FILE_NAME, json.dumps(sets, indent=4, ensure_ascii=False).encode("utf-8"))

    counts["no_symbol"], counts["failed"] = len(no_symbol), len(failed)
    print(f"Download complete. {len(wanted)} sets: {counts['downloaded']} downloaded, "
          f"{counts['kept']} already there, {counts['no_symbol']} without symbol.")
    if no_symbol:
        print(f"No symbol on scryfall (normal): {' '.join(no_symbol)}")
    if failed:
        print(f"Failed, run it again: {' '.join(failed)}")

    import colorize_set_symbol  # Here: it needs numpy, OpenCV and PyMuPDF, the rest of this file doesn't
    colorize_set_symbol.colorize_set_symbols(wanted, force=force)
    return counts


def ask_set_symbols():
    """Interactive download of the set symbols: which sets ('*' = every set)."""
    while True:
        answer = input("Which sets ? (codes separated by spaces, * = every set) : ").strip()
        if answer:
            break
    download_set_symbols(None if answer == "*" else answer.replace(",", " ").split())


def ask_expensions():
    """Interactive download of the cards: which set ('*' = every set), with images or not."""
    # Get the latest set
    list_set()
    exp_code = input(f"Which expension ? : ")
    with_image = input(f"With image (N/y) ? : ")
    if exp_code == "*":
        count = 1
        for exp in expension.keys():
            print(f"Expension '{expension[exp]['name']}' downloading ({count}/{len(expension)})... ")
            card_count = download_expension(exp, with_image == "y")
            print(f"Download complete. {card_count} cards saved.")
            count = count + 1
    else:
        card_count = download_expension(exp_code, with_image == "y")
        print(f"Download complete. {card_count} cards saved.")


def ask_download() -> str:
    """First question: 'symbol' (every card symbol), 'card' (the cards of a set) or 'set' (the set symbols)."""
    choices = {"1": "symbol", "2": "card", "3": "set"}
    for number, choice in choices.items():
        print(f"{number} : {choice}")
    while True:
        answer = input("Your choice ? : ").strip()
        if answer in choices:
            return choices[answer]
        print(f"Answer {' or '.join(choices)}.")


if __name__ == "__main__":
    choice = ask_download()
    if choice == "symbol":
        download_symbols()
    elif choice == "card":
        ask_expensions()
    else:
        ask_set_symbols()
