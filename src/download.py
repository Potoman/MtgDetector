import requests
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import local_paths


HEADERS = {"User-Agent": "mtg-corner-detector/1.0"}

# Project root (<root>), independent of the current working directory.
ROOT = local_paths.ROOT
# The cards (info.json + images): <root>/scryfall unless paths.json says otherwise (see local_paths.py).
SCRYFALL_DIR = local_paths.SCRYFALL_DIR

expension = {}

# Cache of <root>/scryfall/exp_<exp_code>/info.json, indexed by card id.
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
    Save the cards of a set in <root>/scryfall/exp_<exp_code>/info.json (+ their missing images if `image`).
    An existing info.json is reused, not downloaded again: delete it to get the latest data.
    """
    exp_dir = SCRYFALL_DIR / f"exp_{exp_code}"
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
    """Find a card by its scryfall id in <root>/scryfall/exp_<exp_code>/info.json (cached)."""
    if exp_code not in _cards_by_id:
        with open(SCRYFALL_DIR / f"exp_{exp_code}" / "info.json", "r", encoding="utf-8") as fp:
            _cards_by_id[exp_code] = {card["id"]: card for card in json.load(fp)}
    if card_id not in _cards_by_id[exp_code]:
        raise KeyError(f"Card '{card_id}' not found in expension '{exp_code}'")
    return _cards_by_id[exp_code][card_id]


def download_card_images(exp_code: str, card: dict) -> List[str]:
    """
    Download every image of a card (all faces for a double-faced card) that is not already
    in <root>/scryfall/exp_<exp_code>/image/<illustration_id>.jpg.

    :param exp_code: expension code (ex: 'mrd')
    :param card: scryfall json object of the card
    :return: the path of every image of the card available on disk (already present or just
             downloaded), in face order. A face whose download failed is not returned.
    """
    image_dir = SCRYFALL_DIR / f"exp_{exp_code}" / "image"
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
    """download_card_images for a card found by its scryfall id in <root>/scryfall/exp_<exp_code>/info.json."""
    return download_card_images(exp_code, _get_card(exp_code, card_id))


if __name__ == "__main__":
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
