import requests
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple


HEADERS = {"User-Agent": "mtg-corner-detector/1.0"}

# Project root (<root>), independent of the current working directory.
ROOT = Path(__file__).resolve().parent.parent
SCRYFALL_DIR = ROOT / "scryfall"

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
    latest_set = None
    for set_data in sets_data["data"]:
        # You might want to filter by set_type (e.g., "expansion", "core", etc.)
        if set_data["set_type"] in ["expansion", "core"]:
            latest_set = set_data
            expension[set_data["code"]] = set_data
            print(f"> {set_data['code']} : {set_data['name']}")
            #break

    #return latest_set


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
    """Download all card images from a specific set"""
    # Create directory for images
    exp_dir = Path("..") / "scryfall" / f"exp_{exp_code}"
    image_dir = exp_dir / "image"
    os.makedirs(image_dir, exist_ok=True)

    # Set up pagination - Scryfall returns cards in pages
    has_more = True
    next_page_url = f"https://api.scryfall.com/cards/search?q=set:{exp_code}&unique=cards"

    card_count = 0
    page = 1

    cards = []

    # Process each page of results
    while has_more:
        print(f"Processing page {page}...")
        response = requests.get(next_page_url, headers=HEADERS)

        # No card foundfor the expension code.
        if response.status_code == 404:
            has_more = False
            print(f"No cards found for this expension.")
            continue

        # Check for rate limiting or errors
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            time.sleep(1)  # Wait before retry
            continue

        data = response.json()
        cards.extend(data.get("data", []))

        if image:
            # Process each card on the current page
            for card in data.get("data", []):
                card_count += 1
                card_name = card["name"].replace("//", "_").replace(" ", "_")
                image_type = "normal"  # Options include: small, normal, large, png, art_crop, border_crop

                # Some cards have multiple faces (handled by get_card_images)
                for illustration_id, image_url, _ in get_card_images(card, image_type):
                    save_image(image_url, image_dir, f"{illustration_id}")

                # Be nice to the API - don't flood it with requests
                time.sleep(0.1)

        # Check if there are more pages
        has_more = data.get("has_more", False)
        next_page_url = data.get("next_page", None)

        # Respect rate limits - pause between pages
        page += 1
        time.sleep(0.5)

    with open(exp_dir / f'info.json', 'w+') as fp:
        json.dump(cards, fp, indent=4)

    print(f"Downloaded images for {card_count} cards from set {exp_code}")
    return card_count


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


def download_if_not_exist(exp_code: str, card_id: str) -> List[str]:
    """
    Download every image of a card (all faces for a double-faced card) that is not already
    in <root>/scryfall/exp_<exp_code>/image/<illustration_id>.jpg.

    :param exp_code: expension code (ex: 'mrd')
    :param card_id: scryfall id of the card (the 'id' field in info.json)
    :return: the path of every image of the card available on disk (already present or just
             downloaded), in face order. A face whose download failed is not returned.
    """
    image_dir = SCRYFALL_DIR / f"exp_{exp_code}" / "image"
    os.makedirs(image_dir, exist_ok=True)

    paths = []
    for illustration_id, image_url, _ in get_card_images(_get_card(exp_code, card_id)):
        path = image_dir / f"{illustration_id}.jpg"
        if not path.exists():
            downloaded = save_image(image_url, image_dir, f"{illustration_id}")
            # Be nice to the API - don't flood it with requests
            time.sleep(0.1)
            if not downloaded:
                continue
        paths.append(str(path))
    return paths


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
