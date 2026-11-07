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
# <root>/scryfall: the Windows folder (webcam.py). In WSL with paths.json, SCRYFALL_DIR is on the Linux disk instead.
WINDOWS_SCRYFALL_DIR = ROOT / "scryfall"

# The card symbols ({T}, {W}, {W/U}...): <scryfall dir>/symbol/*.svg + symbology.json (see download_symbols).
SYMBOLOGY_URL = "https://api.scryfall.com/symbology"
SYMBOL_DIR_NAME = "symbol"

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

def symbol_dirs() -> List[Path]:
    """
    The folders that get the symbols: SCRYFALL_DIR/symbol, then <root>/scryfall/symbol. In WSL with
    paths.json, the first one is on the Linux disk (~/mtg/scryfall/symbol) and the second one on the
    Windows disk. A single folder when they are the same (on Windows, or without paths.json).
    """
    dirs: List[Path] = []
    for scryfall_dir in (SCRYFALL_DIR, WINDOWS_SCRYFALL_DIR):
        path = scryfall_dir / SYMBOL_DIR_NAME
        if all(path.resolve() != other.resolve() for other in dirs):
            dirs.append(path)
    return dirs


def symbol_file_stem(symbol: str) -> str:
    """
    File name (without .svg) of a symbol, same as mana_cost_gen_dataset.py: '{W/U}' -> 'W-U', '{T}' -> 'T'.
    The other characters that Windows refuses in a file name become '_'.
    """
    return re.sub(r'[<>:"\\|?*]', "_", symbol.strip("{}").replace("/", "-"))


def list_symbols() -> List[dict]:
    """Every card symbol of scryfall (all pages)."""
    symbols = []
    url = SYMBOLOGY_URL
    while url:
        response = requests.get(url, headers={**HEADERS, "Accept": "application/json"}, timeout=30)
        response.raise_for_status()
        page = response.json()
        symbols.extend(page.get("data", []))
        url = page.get("next_page") if page.get("has_more") else None
        if url:
            time.sleep(0.1)  # Respect rate limits
    return symbols


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
    Every scryfall card symbol as an SVG file ({W/U} -> W-U.svg) in each folder of symbol_dirs(), with
    symbology.json: what scryfall says about each symbol (english, colors, mana_value, appears_in_mana_costs...).
    A symbol is downloaded once, then written in both folders. The files already there are kept (force:
    download them again): a symbol missing from one folder only is copied from the other one.
    :return: number of symbols 'downloaded', 'copied' (from the other folder), 'kept' (already everywhere), 'failed'.
    """
    dirs = symbol_dirs()
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

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"{len(with_svg)} symbols for: {', '.join(str(directory) for directory in dirs)}")

    counts = {"downloaded": 0, "copied": 0, "kept": 0, "failed": 0}
    failed = []
    for symbol in with_svg:
        name = f"{symbol_file_stem(symbol['symbol'])}.svg"
        paths = [directory / name for directory in dirs]
        missing = paths if force else [path for path in paths if not path.exists()]
        if not missing:
            counts["kept"] += 1
            continue
        if len(missing) == len(paths):
            content = _download_svg(symbol["svg_uri"])
            time.sleep(0.1)  # Be nice to the API - don't flood it with requests
            if content is None:
                failed.append(symbol["symbol"])
                continue
            action = "downloaded"
        else:
            content = next(path for path in paths if path not in missing).read_bytes()
            action = "copied"
        for path in missing:
            _write_file(path, content)
        counts[action] += 1
        print(f"{action.capitalize()}: {symbol['symbol']} -> {name}")

    for directory in dirs:
        _write_file(directory / "symbology.json", json.dumps(symbols, indent=4, ensure_ascii=False).encode("utf-8"))

    counts["failed"] = len(failed)
    print(f"Download complete. {len(with_svg)} symbols: {counts['downloaded']} downloaded, "
          f"{counts['copied']} copied from the other folder, {counts['kept']} already there.")
    if without_svg:
        print(f"No SVG on scryfall, skipped: {' '.join(without_svg)}")
    if failed:
        print(f"Failed, run it again: {' '.join(failed)}")
    return counts


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
    """First question: 'set' (the cards of a set) or 'symbol' (every card symbol)."""
    choices = {"1": "set", "2": "symbol"}
    for number, choice in choices.items():
        print(f"{number} : {choice}")
    while True:
        answer = input("Your choice ? : ").strip()
        if answer in choices:
            return choices[answer]
        print(f"Answer {' or '.join(choices)}.")


if __name__ == "__main__":
    if ask_download() == "symbol":
        download_symbols()
    else:
        ask_expensions()
