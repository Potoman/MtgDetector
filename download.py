import requests
import json
import os
import time


expension = {}


def list_set():
    """Get information about the most recent MTG set"""
    sets_url = "https://api.scryfall.com/sets"
    response = requests.get(sets_url)
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


def download_set_images(exp_code):
    """Download all card images from a specific set"""
    # Create directory for images
    image_dir = f"mtg_images_{exp_code}"
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
        response = requests.get(next_page_url)

        # Check for rate limiting or errors
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            time.sleep(1)  # Wait before retry
            continue

        data = response.json()
        cards.extend(data.get("data", []))

        # Process each card on the current page
        for card in data.get("data", []):
            card_count += 1
            card_name = card["name"].replace("//", "_").replace(" ", "_")
            image_type = "normal"  # Options include: small, normal, large, png, art_crop, border_crop

            # Some cards have multiple faces
            if "image_uris" in card:
                # Single-faced card
                image_url = card["image_uris"].get(image_type)
                save_image(image_url, image_dir, f"{card['illustration_id']}")
            elif "card_faces" in card:
                # Double-faced card
                for i, face in enumerate(card["card_faces"]):
                    if "image_uris" in face:
                        image_url = face["image_uris"].get(image_type)
                        save_image(image_url, image_dir, f"{face['illustration_id']}")

            # Be nice to the API - don't flood it with requests
            time.sleep(0.1)

        # Check if there are more pages
        has_more = data.get("has_more", False)
        next_page_url = data.get("next_page", None)

        # Respect rate limits - pause between pages
        page += 1
        time.sleep(0.5)

    with open(f'result_{exp_code}.json', 'w+') as fp:
        json.dump(cards, fp, indent=4)

    print(f"Downloaded images for {card_count} cards from set {exp_code}")
    return card_count


def save_image(image_url, directory, filename):
    """Download and save an image"""
    if not image_url:
        return False

    try:
        response = requests.get(image_url)
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





if __name__ == "__main__":
    # Get the latest set
    list_set()
    exp_code = input(f"Which expension ? : ")
    card_count = download_set_images(exp_code)
    print(f"Download complete. {card_count} cards saved.")




