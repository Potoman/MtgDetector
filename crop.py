import json

from PIL import Image
import os
import random
from typing import Dict, List, Tuple


def get_crop_exp(exp_code: str, rarity: str, color: str, illustration_id: str) -> Tuple[Image, Dict[str, str]]:
    input_folder = f"mtg_images_{exp_code}"
    output_folder = os.path.join("exp_code")
    os.makedirs(output_folder, exist_ok=True)
    height = 80
    width = 80
    x = 430
    y = 400
    assert (x + width / 2 < 488)

    input_path = os.path.join(input_folder, illustration_id + '.jpg')

    # Open and crop the image
    image = Image.open(input_path)
    crop_box = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)  # Example coordinates
    cropped_image = image.crop(crop_box)

    return cropped_image, {'exp_code': exp_code, 'rarity': rarity, 'color': color}


def crop_image(exp_code: str, labels: Dict, rarity: str, color: str, illustration_id: str):
    input_folder = f"mtg_images_{exp_code}"
    output_folder = os.path.join("exp_code")
    os.makedirs(output_folder, exist_ok=True)
    height = 80
    width = 80
    x = 430
    y = 400
    assert(x + width / 2 < 488)

    input_path = os.path.join(input_folder, illustration_id + '.jpg')
    output_path = os.path.join(output_folder, f"{exp_code}_{rarity}_{color}.png")

    if not os.path.exists(output_path):
        for index in range(5):
            while True:
                rnd_x = random.randrange(0, 488, 1)
                rnd_y = random.randrange(0, 680, 1)
                if (((430 - 80) < rnd_x < (430 + 80)) and
                        ((400 - 80) < rnd_y < (400 + 80)) or
                        (rnd_x < 40) or
                        ((488 - 40) < rnd_x) or
                        (rnd_y < 40) or
                        ((680 - 40) < rnd_y)):
                    continue
                break
            output_path_error = os.path.join(output_folder, f"{exp_code}_{rarity}_{color}_error_{index}.png")
            image = Image.open(input_path)
            crop_box = (rnd_x - width / 2, rnd_y - height / 2, rnd_x + width / 2, rnd_y + height / 2)  # Example coordinates
            cropped_image = image.crop(crop_box)
            cropped_image.save(output_path_error)
            labels[output_path_error.replace('\\', '/')] = {'exp_code': '', 'rarity': '', 'color': ''}

    # Open and crop the image
    image = Image.open(input_path)
    crop_box = (x - width / 2, y - height / 2, x + width / 2, y + height / 2)  # Example coordinates
    cropped_image = image.crop(crop_box)

    # Save the cropped image
    cropped_image.save(output_path)

    labels[output_path.replace('\\', '/')] = {'exp_code': exp_code, 'rarity': rarity, 'color': color}

    print(f"Processed: {illustration_id}")


def crop_exp_code(exp_code: str, labels: Dict):
    with open(f'result_{exp_code}.json', 'r') as file:
        cards = json.load(file)

        for card in cards:
            rarity = card['rarity']

            # Some cards have multiple faces
            if "image_uris" in card:
                # Single-faced card
                colors = card['colors']
                color = "none" if not colors else "multi" if len(colors) > 1 else colors[0]
                crop_image(exp_code, labels, rarity, color, f"{card['illustration_id']}")
            elif "card_faces" in card:
                # Double-faced card
                for i, face in enumerate(card["card_faces"]):
                    if "image_uris" in face:
                        colors = face['colors']
                        color = "none" if not colors else "multi" if len(colors) > 1 else colors[0]
                        crop_image(exp_code, labels, rarity, color, f"{face['illustration_id']}")


def crop_exp_codes(exp_codes: List[str]):
    labels = {}
    for exp_code in exp_codes:
        crop_exp_code(exp_code, labels)
    with open('labels.json', 'w') as output:
        json.dump(labels, output, indent=4)


def get_crop_and_labels(exp_code: str) -> List[Tuple[Image, Dict[str, str]]]:
    crop_and_labels: List[Tuple[Image, Dict[str, str]]] = []
    with open(f'result_{exp_code}.json', 'r') as file:
        cards = json.load(file)

        for card in cards:
            rarity = card['rarity']

            # Some cards have multiple faces
            if "image_uris" in card:
                # Single-faced card
                colors = card['colors']
                color = "none" if not colors else "multi" if len(colors) > 1 else colors[0]
                crop_and_labels.append(get_crop_exp(exp_code, rarity, color, f"{card['illustration_id']}"))
            elif "card_faces" in card:
                # Double-faced card
                for i, face in enumerate(card["card_faces"]):
                    if "image_uris" in face:
                        colors = face['colors']
                        color = "none" if not colors else "multi" if len(colors) > 1 else colors[0]
                        crop_and_labels.append(get_crop_exp(exp_code, rarity, color, f"{face['illustration_id']}"))
    return crop_and_labels


def crop():
    exp_code = input(f"Which exp_code ? : ")
    labels = {}
    crop_exp_code(exp_code, labels)
    pass


if __name__ == "__main__":
    crop_exp_codes(['mrd', 'znr'])