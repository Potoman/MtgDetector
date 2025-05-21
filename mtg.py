import json
from typing import List


def get_illustrations_ids(exp_code: str) -> List[str]:
    illustration_ids = []
    with open(f'result_{exp_code}.json', 'r') as file:
        cards = json.load(file)
        for card in cards:
            # Some cards have multiple faces
            if "image_uris" in card:
                # Single-faced card
                illustration_ids.append(f"{card['illustration_id']}")
            elif "card_faces" in card:
                # Double-faced card
                for i, face in enumerate(card["card_faces"]):
                    if "image_uris" in face:
                        illustration_ids.append(f"{face['illustration_id']}")
    return illustration_ids
