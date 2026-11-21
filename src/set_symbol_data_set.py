"""
Dataset of the set symbol detector: generated images to train, real cards to validate.

    python3 src/set_symbol_data_set.py [n] [--val PERCENT] [--nosym PERCENT] [--force]   (asked if not given)

Also built by the 'set_gen' training of training_set_symbol.py, which trains on it.
The detector finds the set symbol (expansion symbol), its box, and the rarity shown by its color.

n samples in <root>/runs/detect/set/gen/<n>_val<VAL>_nosym<NOSYM>/,
ex: gen/80000_val2.5_nosym5/: VAL percent of them are real cards for the validation, the others are
generated for the training, NOSYM percent of each split has no set symbol.
- The symbols: the PNG files of scryfall/set_png/, set_<icon>_<rarity>.png, made by colorize_set_symbol.py
  with its colored SVG (run by download.py, choice 3: set), named after the icon of the set in scryfall
  (sets.json "icon_svg_uri": the promos and tokens of a set print its symbol, ex: tdsk -> 'dsk'). Each PNG
  has the symbol at its final size (SYMBOL_HEIGHT = colorize_set_symbol.PNG_SYMBOL_HEIGHT: 27 px high, its
  outline included): it is cut out of the PNG and pasted as it is, not resized, its colors not changed.
  The training changes the brightness, the saturation, the gamma, the blur... of every image (see
  training_set_symbol.py), as for the mana cost.
- One class per set and rarity, '<icon>_<rarity>' (ex: 'dsk_rare', from set_dsk_rare.png), see
  class_name: common, uncommon, rare for the sets released since Exodus (EXODUS_DATE, the first rarity
  colors), mythic since Shards of Alara (ALARA_DATE). From Arabian Nights to Stronghold (before Exodus) a
  symbol has one look for every rarity: one class per set, '<icon>' (ex: 'arn', from
  set_arn_common.png, the 3 files are the same). Only the classes seen on the real cards on disk
  (VAL_SET_TYPES, sets released since Arabian Nights, the first set with a symbol), so that every class
  is validated. The cards of another rarity (special, bonus...) are not used. NO_SYMBOL_SETS print no
  symbol (checked on the scans): never a class, never in the validation.
- No symbol: the cards released before Arabian Nights (FIRST_SET_DATE: Alpha, Beta, Unlimited, the "old
  cards") print no symbol. Their symbol zone (SYMBOL_ZONE, the window VAL_RECT) is the image without
  symbol: real old cards in the validation, other old cards (never the same card names) in the training.
- train: n_train = n - n_val generated images of IMAGE_WIDTH x IMAGE_HEIGHT (150 x 150), the size of the window:
  - NOSYM percent of them: the empty symbol zone of an old card, no box.
  - the others: one symbol, the same number of images for every set (its icon: 'atq' or 'ice' as
    much as 'mrd'), split evenly between the rarities of the set ('mrd_common', 'mrd_uncommon',
    'mrd_rare': a third each), rounded (see balanced_counts), in random order, on a background
    (BACKGROUNDS):
    - "old_symbol_zone": the empty symbol zone of an old card (as the images without symbol: the
      detector has to look at the symbol itself, not at the old frame),
    - "card": a random part of a real card, never touching SYMBOL_ZONE (where its set symbol is
      printed): no real set symbol in a generated image,
    - "type_line": the left part of the type line of a card (same height as the window, left of SYMBOL_ZONE),
    - "color": one random color,
    - "gradient": a gradient between two random colors, in a random direction.
    Text on TEXT_RATIO of them (not on an old symbol zone): a few lines of card names, each line black or
    white. The symbol: its PNG, at a random place fully inside the image, always SYMBOL_HEIGHT (27 px)
    high, its outline included (1.5 px, see colorize_set_symbol.OUTLINE), its width following its shape
    (16 to 66 px). The training zooms the images (scale +/-20%): 22 to 32 px high, outline 1.2 to 1.8 px. One box per
    symbol (its outline included), its class '<icon>_<rarity>' (or '<icon>').
  The cards of the validation are never used as backgrounds.
- val: n_val = round(n * VAL / 100) real cards (frames FRAMES), the window VAL_RECT (also the search
  window of the detector), without box: the place of the symbol on a real card is not known, the
  validation only checks the class read (training_set_symbol.py reads the "class" of images.json).
  NOSYM percent of them: old cards (no symbol). The others: as the training, the same number of cards for
  every set (the old sets count as much as the new ones), split evenly between its rarities, rounded. A
  set with fewer real cards than its share gives them all, the rest is split between the other sets; a
  rarity with fewer cards than its share gives them all, the rest goes to the other rarities of its set.
  Augmented as the training images (val_augmentation, given by training_set_symbol.py: its hsv_h,
  hsv_s, hsv_v and augmentations): one draw per card, seeded (SEED), made once when the dataset is
  built (Ultralytics never augments the validation). None: the real cards as they are.

Files: data.yaml, classes.json, images/{train,val}/*.png + labels/{train,val}/*.txt (YOLO format: empty for
the validation and for the images without symbol), images.json (each image: "class", null = no symbol,
"box", null in the validation, "source"), report.json (images per class), check/ (first images of each
split, with their box in train), and dataset.stamp: the settings of the dataset, written last. The dataset
is not built again while dataset.stamp matches (--force: always). New card images or symbols downloaded
later change the classes only with --force.
"""
import argparse
import functools
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image, ImageDraw, ImageFont

import colorize_set_symbol
import download
import local_paths
import training
import training_mana_cost as mana

VERSION = 11  # Change it when the generation changes: every dataset is built again
SEED = 0      # Same settings -> same dataset

Box = training.Box

# --- Sets ---
FIRST_SET_DATE = "1993-12-17"   # Arabian Nights, the first set with a set symbol: the older cards print none
EXODUS_DATE = "1998-06-15"      # Exodus: the first set with rarity colors (common, uncommon, rare)
ALARA_DATE = "2008-10-03"       # Shards of Alara: the first set with mythic rares
RARITIES = ["common", "uncommon", "rare", "mythic"]
OLD_SYMBOL_RARITY = "common"    # Before Exodus, the class '<icon>' is drawn from set_<icon>_common.svg
# Sets printed without set symbol (checked on the scans: 3ed, 4ed, 5ed, fbb, 4bb; chr, ren, rin, sum: same cards).
NO_SYMBOL_SETS = ["3ed", "4ed", "5ed", "fbb", "4bb", "chr", "ren", "rin", "sum"]
# Set types whose cards print the symbol of their set at the usual place (not: The List, Secret Lair,
# tokens, art series, promos, digital sets, oversized cards...). Also the old cards used: core sets.
VAL_SET_TYPES = ["expansion", "core", "masters", "draft_innovation", "commander", "duel_deck", "from_the_vault",
                 "starter", "funny", "eternal", "arsenal", "spellbook", "premium_deck"]
FRAMES = ["1993", "1997", "2003", "2015"]
# Prints whose set symbol is elsewhere (or missing): never used, not even as a background (it would
# bring a second symbol). Flip cards (Kamigawa), battles (printed sideways), borderless, full art,
# textless, special frames.
EXCLUDED_LAYOUTS = ["flip"]
EXCLUDED_FRAME_EFFECTS = ["showcase", "inverted", "etched", "fullart", "extendedart"]
EXCLUDED_PROMO_TYPES = ["boosterfun"]

# --- Window ---
IMAGE_WIDTH, IMAGE_HEIGHT = 150, 150
# Window of the real cards (validation) and search window of the detector: the right end of the
# type line, on every frame: x 338-488, y 325-475 of a CARD_WIDTH x CARD_HEIGHT card.
VAL_RECT = training.Rect(center_x=training.CARD_WIDTH - IMAGE_WIDTH / 2, center_y=400,
                         height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
SYMBOL_ZONE = (int(VAL_RECT.center_x - IMAGE_WIDTH / 2), int(VAL_RECT.center_y - IMAGE_HEIGHT / 2),
               int(VAL_RECT.center_x + IMAGE_WIDTH / 2), int(VAL_RECT.center_y + IMAGE_HEIGHT / 2))

# --- Symbols ---
# Height of every generated symbol, its outline included (pixels), its width following its shape: the
# size of the symbol in its PNG (the PNG is pasted as it is).
SYMBOL_HEIGHT = colorize_set_symbol.PNG_SYMBOL_HEIGHT

# --- Backgrounds of the generated images with a symbol (see the top of the file) ---
BACKGROUNDS = {"old_symbol_zone": 0.20, "card": 0.28, "type_line": 0.20, "color": 0.16, "gradient": 0.16}
TYPE_LINE_SHIFT = 16    # "type_line": up to this many pixels above / below the window
TEXT_RATIO = 0.15
TEXT_LINES = (1, 3)
TEXT_SIZES = (10, 22)
TEXT_FONT_DIRS = ["/mnt/c/Windows/Fonts", "/usr/share/fonts/truetype/dejavu"]
TEXT_FONT_FILES = ["timesbd.ttf", "times.ttf", "georgiab.ttf", "georgia.ttf", "palab.ttf", "cambriab.ttf",
                   "DejaVuSerif-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSans-Bold.ttf"]
HERSHEY_FONTS = [cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]

# --- Splits ---
VAL_PERCENT = 2.5           # Share of the samples for the validation (percent), proposed when it is asked
NO_SYMBOL_PERCENT = 5.0     # Share of each split without symbol (percent), proposed when it is asked
MIN_SAMPLES = 40

# --- Output ---
OUTPUT_DIR = training.DETECT_OUTPUT_DIR / "set" / "gen"
STAMP_FILE = "dataset.stamp"
CHECK_DIR = "check"
CHECK_IMAGES = 50       # First images of each split saved (zoomed, with their box) in check/
CHECK_ZOOM = 3


# ---------------------------------------------------------------------------
# Sets, classes and symbols
# ---------------------------------------------------------------------------

def symbol_dir() -> Path:
    return download.symbol_dir(download.SET_DIR_NAME)


def load_sets() -> Dict[str, dict]:
    """The scryfall sets (scryfall/set/sets.json), by code."""
    path = symbol_dir() / download.SETS_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"No {path}: download the set symbols first (python3 src/download.py, choice 3: set).")
    with open(path, "r", encoding="utf-8") as fp:
        return {set_data["code"]: set_data for set_data in json.load(fp)}


def icon_of(set_data: dict) -> str:
    """The symbol printed on the cards of a set: the name of its scryfall icon (tdsk -> 'dsk')."""
    uri = re.sub(r"\?.*$", "", set_data.get("icon_svg_uri") or "")
    return uri.rsplit("/", 1)[-1][:-len(".svg")] if uri.endswith(".svg") else set_data["code"]


def class_name(icon: str, rarity: Optional[str]) -> str:
    """'<icon>_<rarity>' (ex: 'dsk_rare'), '<icon>' for a set before Exodus (rarity None)."""
    return f"{icon}_{rarity}" if rarity else icon


def parse_class(name: str) -> Tuple[str, Optional[str]]:
    """Inverse of class_name: 'dsk_rare' -> ('dsk', 'rare'), 'arn' -> ('arn', None)."""
    icon, _, rarity = name.rpartition("_")
    return (icon, rarity) if icon and rarity in RARITIES else (name, None)


def rarities_of(released_at: Optional[str]) -> List[Optional[str]]:
    """The rarities with their own symbol color in a set released on that date ('YYYY-MM-DD'): [None]
    before Exodus (one look), common to rare, mythic since Shards of Alara (unknown: all)."""
    if released_at and released_at < EXODUS_DATE:
        return [None]
    return [rarity for rarity in RARITIES if rarity != "mythic" or not released_at or released_at >= ALARA_DATE]


def png_dir() -> Path:
    return download.symbol_dir(colorize_set_symbol.PNG_DIR_NAME)


def symbol_file(label: str) -> Path:
    """The PNG of a class (made by colorize_set_symbol.py): set_<icon>_<rarity>.png, set_<icon>_common.png
    for a set before Exodus."""
    icon, rarity = parse_class(label)
    return png_dir() / colorize_set_symbol.png_file_name(icon, rarity or OLD_SYMBOL_RARITY)


def drawing_stamps() -> List[str]:
    """
    How the PNG of png_dir() were made: their png_stamp (colorize_set_symbol, text chunk PNG_TEXT_KEY: the
    paint, the drawing version, the PNG size), each one once. A new coloring changes them (in the <hash>
    of the training). Cheap: reads the start of each file.
    """
    stamps = set()
    folder = png_dir()
    if folder.is_dir():
        for path in folder.glob("set_*_*.png"):
            try:
                with Image.open(path) as image:
                    stamps.add(image.info.get(colorize_set_symbol.PNG_TEXT_KEY) or "unknown")
            except (OSError, ValueError):
                stamps.add("unreadable")
    return sorted(stamps)


def load_symbol(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    The symbol of a PNG of colorize_set_symbol.py, cut out of it, ready to paste as it is: (BGR premultiplied
    by the alpha float32 0-255 H x W x 3, alpha 0-1 H x W), SYMBOL_HEIGHT high (its outline included).
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # BGRA, not premultiplied
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Can't read {path} with transparency")
    ys, xs = np.nonzero(image[:, :, 3])
    if len(xs) == 0:
        raise ValueError(f"Empty symbol: {path}")
    image = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    alpha = image[:, :, 3] / 255
    return image[:, :, :3] * alpha[:, :, None], alpha


def paste_symbol(background_bgr: np.ndarray, layer: np.ndarray, alpha: np.ndarray,
                 rng: random.Random) -> Tuple[np.ndarray, Box]:
    """The symbol at a random place fully inside the image. :return: (image BGR, box of the symbol)."""
    height, width = alpha.shape
    ys, xs = np.nonzero(alpha > 0.25)
    if len(xs) == 0:
        raise ValueError("invisible symbol")
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if bx1 - bx0 > IMAGE_WIDTH or by1 - by0 > IMAGE_HEIGHT:
        raise ValueError("symbol bigger than the image")
    x = rng.randint(-bx0, IMAGE_WIDTH - bx1)
    y = rng.randint(-by0, IMAGE_HEIGHT - by1)
    image = background_bgr.astype(np.float32)
    # Part of the layer inside the image (its faint edge may go out).
    ix0, iy0, ix1, iy1 = max(0, x), max(0, y), min(IMAGE_WIDTH, x + width), min(IMAGE_HEIGHT, y + height)
    part_alpha = alpha[iy0 - y:iy1 - y, ix0 - x:ix1 - x, None]
    part_bgr = layer[iy0 - y:iy1 - y, ix0 - x:ix1 - x]
    image[iy0:iy1, ix0:ix1] = part_bgr + image[iy0:iy1, ix0:ix1] * (1 - part_alpha)
    box = (int(x + bx0), int(y + by0), int(x + bx1), int(y + by1))
    return np.clip(image + 0.5, 0, 255).astype(np.uint8), box


# ---------------------------------------------------------------------------
# Backgrounds and text
# ---------------------------------------------------------------------------

def _random_color(rng: random.Random) -> np.ndarray:
    return np.array([rng.randint(0, 255) for _ in range(3)], np.float32)


def color_background(rng: random.Random) -> np.ndarray:
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), _random_color(rng), np.uint8)


def gradient_background(rng: random.Random) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH].astype(np.float32)
    position = xs * np.cos(angle) + ys * np.sin(angle)
    position = ((position - position.min()) / (position.max() - position.min()))[:, :, None]
    first, second = _random_color(rng), _random_color(rng)
    return np.clip(first * (1 - position) + second * position + 0.5, 0, 255).astype(np.uint8)


def _touches_symbol_zone(x: int, y: int) -> bool:
    zx0, zy0, zx1, zy1 = SYMBOL_ZONE
    return x < zx1 and x + IMAGE_WIDTH > zx0 and y < zy1 and y + IMAGE_HEIGHT > zy0


def card_background(card_bgr: np.ndarray, kind: str, rng: random.Random) -> np.ndarray:
    """A part of a real card that doesn't touch SYMBOL_ZONE ("card": anywhere, "type_line": left of it)."""
    if kind == "type_line":
        x = rng.randint(0, SYMBOL_ZONE[0] - IMAGE_WIDTH)
        y = SYMBOL_ZONE[1] + rng.randint(-TYPE_LINE_SHIFT, TYPE_LINE_SHIFT)
    else:
        while True:
            x = rng.randint(0, training.CARD_WIDTH - IMAGE_WIDTH)
            y = rng.randint(0, training.CARD_HEIGHT - IMAGE_HEIGHT)
            if not _touches_symbol_zone(x, y):
                break
    return card_bgr[y:y + IMAGE_HEIGHT, x:x + IMAGE_WIDTH].copy()


def text_fonts() -> List[Path]:
    """The TEXT_FONT_FILES found in TEXT_FONT_DIRS (empty: the OpenCV Hershey fonts are used)."""
    return [Path(folder) / name for folder in TEXT_FONT_DIRS for name in TEXT_FONT_FILES
            if (Path(folder) / name).is_file()]


@functools.lru_cache(maxsize=None)
def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def draw_text(image_bgr: np.ndarray, names: List[str], fonts: List[Path], rng: random.Random) -> np.ndarray:
    """TEXT_LINES lines of card names at random places (cut at the edges), each line black or white."""
    lines = []
    for _ in range(rng.randint(*TEXT_LINES)):
        text = " ".join(rng.choice(names) for _ in range(rng.randint(1, 2)))
        size = rng.randint(*TEXT_SIZES)
        x = rng.randint(-IMAGE_WIDTH // 2, IMAGE_WIDTH - 2 * size)
        y = rng.randint(-size // 3, IMAGE_HEIGHT - size)
        lines.append((text, size, x, y, (0, 0, 0) if rng.random() < 0.5 else (255, 255, 255)))
    if not fonts:
        image = image_bgr.copy()
        for text, size, x, y, color in lines:
            font = rng.choice(HERSHEY_FONTS)
            scale = 0.7 * size / cv2.getTextSize("H", font, 1.0, 1)[0][1]
            cv2.putText(image, text, (x, y + size), font, scale, color, 1, cv2.LINE_AA)
        return image
    image = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]))
    draw = ImageDraw.Draw(image)
    for text, size, x, y, color in lines:
        draw.text((x, y), text, fill=color, font=_font(rng.choice(fonts), size))
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])


# ---------------------------------------------------------------------------
# Real cards
# ---------------------------------------------------------------------------

@dataclass
class CardImage:
    path: Path
    exp_code: str
    card: dict
    frame: str
    icon: Optional[str]     # The symbol printed on the card (None: an old card or a set of NO_SYMBOL_SETS)
    label: Optional[str]    # Its class (class_name), None: no symbol, or a rarity that is not a class
    for_val: bool           # Its set prints its symbol at the usual place (VAL_SET_TYPES, not digital)
    old: bool = False       # Released before Arabian Nights (FIRST_SET_DATE, VAL_SET_TYPES): no symbol


def list_card_images(sets: Dict[str, dict]) -> List[CardImage]:
    """
    The card images on disk, frames FRAMES, layouts and filters of training_mana_cost.py (no planeswalker:
    their symbol is elsewhere), standard prints only (standard_print), one per illustration and set.
    """
    accept = {"frame": FRAMES, "layout": mana.LAYOUTS}
    images, seen = [], set()
    for exp_code in tqdm.tqdm(training.list_exp_codes(), desc="Read info.json", unit="expansion"):
        set_data = sets.get(exp_code)
        if set_data is None:
            continue
        exp_dir = local_paths.EXP_DIR / f"exp_{exp_code}"
        image_dir = exp_dir / "image"
        on_disk = set(os.listdir(image_dir)) if image_dir.is_dir() else set()
        if not on_disk:
            continue
        usual = set_data.get("set_type") in VAL_SET_TYPES and not set_data.get("digital")
        before_symbols = (set_data.get("released_at") or "9999") < FIRST_SET_DATE
        old = before_symbols and usual
        icon = None if before_symbols or exp_code in NO_SYMBOL_SETS else icon_of(set_data)
        for_val = bool(icon) and usual
        # The rarity colors of the symbol follow the release date of the set that owns it.
        rarities = rarities_of((sets.get(icon) or set_data).get("released_at")) if icon else []
        with open(exp_dir / "info.json", "r", encoding="utf-8") as fp:
            cards = json.load(fp)
        for card in cards:
            for illustration_id, _, face in download.get_card_images(card):
                path = image_dir / f"{illustration_id}.jpg"
                if f"{illustration_id}.jpg" not in on_disk or path in seen:  # A reprint elsewhere: its own symbol
                    continue
                if not training._is_accepted(card, face, accept) or training._is_filtered(card, face, mana.FILTERS) \
                        or not standard_print(card, face):
                    continue
                seen.add(path)
                rarity = card.get("rarity")
                label = None
                if icon and rarities == [None]:
                    label = class_name(icon, None)
                elif icon and rarity in rarities:
                    label = class_name(icon, rarity)
                images.append(CardImage(path, exp_code, card, card.get("frame"), icon, label, for_val, old))
    return images


def standard_print(card: dict, face: dict) -> bool:
    """True if the face has its set symbol at the usual place of its frame (see EXCLUDED_LAYOUTS...)."""
    return (card.get("layout") not in EXCLUDED_LAYOUTS
            and card.get("border_color") != "borderless" and not card.get("full_art") and not card.get("textless")
            and not set(card.get("frame_effects") or []) & set(EXCLUDED_FRAME_EFFECTS)
            and not set(card.get("promo_types") or []) & set(EXCLUDED_PROMO_TYPES)
            and "Battle" not in (face.get("type_line") or card.get("type_line") or ""))


def read_card(path: Path) -> Optional[np.ndarray]:
    """A card image resized to CARD_WIDTH x CARD_HEIGHT, None if unreadable."""
    card_bgr = cv2.imread(str(path))
    if card_bgr is None:
        print(f"  Can't read {path}")
        return None
    return cv2.resize(card_bgr, (training.CARD_WIDTH, training.CARD_HEIGHT), interpolation=cv2.INTER_LINEAR)


def real_window(card_bgr: np.ndarray) -> np.ndarray:
    """The symbol zone of a card (SYMBOL_ZONE = VAL_RECT): the image the detector reads."""
    x0, y0, x1, y1 = SYMBOL_ZONE
    return card_bgr[y0:y1, x0:x1].copy()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Sizes:
    """The number of images of a dataset of n samples."""
    train: int
    val: int
    train_nothing: int  # Images without symbol in the training (empty symbol zones of old cards)
    val_nothing: int    # Old cards in the validation


def sizes(n: int, val_percent: float, no_symbol_percent: float) -> Sizes:
    n_val = round(n * val_percent / 100)
    n_train = n - n_val
    return Sizes(n_train, n_val, round(n_train * no_symbol_percent / 100), round(n_val * no_symbol_percent / 100))


def dataset_dir(n: int, val_percent: float, no_symbol_percent: float) -> Path:
    return OUTPUT_DIR / f"{n}_val{val_percent:g}_nosym{no_symbol_percent:g}"


def classes_and_cards() -> Tuple[List[str], List[CardImage], Dict[str, dict]]:
    """The classes (set and rarity seen on real cards of the validation, with a PNG), every card
    image, the sets."""
    sets = load_sets()
    cards = list_card_images(sets)
    with_cards = {card.label for card in cards if card.label and card.for_val}
    missing = sorted(label for label in with_cards if not symbol_file(label).exists())
    # Not made yet: the black SVG of scryfall is there, colorize_set_symbol.py has not made the colored PNG.
    not_drawn = [label for label in missing
                 if (symbol_dir() / download.set_symbol_file_name(parse_class(label)[0])).exists()]
    if not_drawn:
        print(f"WARNING: {len(not_drawn)} classes skipped, their PNG is not made in {png_dir()} "
              f"({', '.join(symbol_file(label).name for label in not_drawn[:8])}{'...' if len(not_drawn) > 8 else ''}). "
              f"Draw them: python3 src/colorize_set_symbol.py")
    if len(missing) > len(not_drawn):
        print(f"{len(missing) - len(not_drawn)} classes skipped: scryfall has no symbol for their icon.")
    return sorted(with_cards - set(missing)), cards, sets


def symbols_hash(classes: List[str]) -> str:
    """Hash of the PNG files of the classes (their content)."""
    digest = hashlib.sha256()
    for label in classes:
        path = symbol_file(label)
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def val_augmentation_of(hsv_h: float, hsv_s: float, hsv_v: float,
                        augmentations: List[Tuple[str, dict]]) -> dict:
    """The augmentation of the validation (json): the hsv gains and the albumentations of a training."""
    return json.loads(json.dumps({"hsv": [hsv_h, hsv_s, hsv_v], "augmentations": augmentations}))


class _ValAugmenter:
    """
    Changes each real card of the validation as the training changes its images (Ultralytics v8_transforms,
    without the spatial ones): the albumentations first, on the BGR image as Ultralytics gives it, then
    RandomHSV (its LUT, gains drawn in +/- hsv). Seeded: the same dataset settings, the same images.
    """

    def __init__(self, augmentation: dict):
        self.hsv = np.array(augmentation["hsv"], dtype=np.float64)
        self.rng = np.random.default_rng(SEED)
        self.compose = None
        if augmentation["augmentations"]:
            import albumentations as A  # Only needed here (pip install albumentations)
            self.compose = A.Compose([getattr(A, name)(**params) for name, params in augmentation["augmentations"]],
                                     seed=SEED)

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.compose is not None:
            image_bgr = self.compose(image=image_bgr)["image"]
        gains = self.rng.uniform(-1, 1, 3) * self.hsv
        if not gains.any():
            return image_bgr
        x = np.arange(0, 256, dtype=np.float64)
        lut_hue = ((x + gains[0] * 180) % 180).astype(np.uint8)
        lut_sat = np.clip(x * (gains[1] + 1), 0, 255).astype(np.uint8)
        lut_val = np.clip(x * (gains[2] + 1), 0, 255).astype(np.uint8)
        lut_sat[0] = 0  # As Ultralytics: pure white keeps no color
        lut = np.stack((lut_hue, lut_sat, lut_val), -1).reshape(1, 256, 3)
        return cv2.cvtColor(cv2.LUT(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV), lut), cv2.COLOR_HSV2BGR)


def dataset_stamp(n: int, val_percent: float, no_symbol_percent: float, classes: List[str],
                  val_augmentation: Optional[dict] = None) -> dict:
    """What the dataset depends on (dataset.stamp), as written in json."""
    stamp = {
        "version": VERSION, "n": n, "seed": SEED, "val_percent": val_percent, "no_symbol_percent": no_symbol_percent,
        "val_augmentation": val_augmentation,
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT], "val_rect": list(VAL_RECT),
        "classes": classes, "symbols_hash": symbols_hash(classes), "old_symbol_rarity": OLD_SYMBOL_RARITY,
        "first_set_date": FIRST_SET_DATE, "no_symbol_sets": NO_SYMBOL_SETS, "val_set_types": VAL_SET_TYPES,
        "frames": FRAMES, "excluded": [EXCLUDED_LAYOUTS, EXCLUDED_FRAME_EFFECTS,
                                                                      EXCLUDED_PROMO_TYPES, "borderless", "full_art", "textless", "battle"],
        "symbol": {"height": SYMBOL_HEIGHT, "files": "set_png"},
        "rarities": {"exodus": EXODUS_DATE, "alara": ALARA_DATE, "rarities": RARITIES},
        "backgrounds": {"kinds": BACKGROUNDS, "type_line_shift": TYPE_LINE_SHIFT},
        "text": {"ratio": TEXT_RATIO, "lines": list(TEXT_LINES), "sizes": list(TEXT_SIZES),
                 "fonts": [font.name for font in text_fonts()] or "hershey"},
        "cards": {"layouts": mana.LAYOUTS, "filters": mana.FILTERS},
    }
    return json.loads(json.dumps(stamp))  # Tuples -> lists: equal to the stamp read back


class _Writer:
    """Writes the images and labels of a split (YOLO format) and their check images."""

    def __init__(self, root: Path, class_to_index: Dict[str, int]):
        self.root, self.class_to_index = root, class_to_index
        self.entries: List[dict] = []
        self.counts: Dict[str, Counter] = {}

    def write(self, split: str, image_bgr: np.ndarray, symbol: Optional[str], box: Optional[Box], source: str):
        """symbol: the class (None: no symbol). box: its box (None: no box, the validation)."""
        counts = self.counts.setdefault(split, Counter())
        index = counts["_images"]
        name = f"{index:06d}"
        image_dir, label_dir = self.root / "images" / split, self.root / "labels" / split
        if index == 0:
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / f"{name}.png"), image_bgr)
        with open(label_dir / f"{name}.txt", "w", encoding="utf-8") as fp:
            if symbol is not None and box is not None:  # Empty file: no box
                x0, y0, x1, y1 = box
                fp.write(f"{self.class_to_index[symbol]} {(x0 + x1) / 2 / IMAGE_WIDTH:.6f} "
                         f"{(y0 + y1) / 2 / IMAGE_HEIGHT:.6f} {(x1 - x0) / IMAGE_WIDTH:.6f} {(y1 - y0) / IMAGE_HEIGHT:.6f}\n")
        self.entries.append({"file": f"images/{split}/{name}.png", "class": symbol,
                             "box": list(box) if box else None, "source": source})
        counts["_images"] += 1
        counts[symbol or "_nothing"] += 1
        if index < CHECK_IMAGES:
            check = cv2.resize(image_bgr, None, fx=CHECK_ZOOM, fy=CHECK_ZOOM, interpolation=cv2.INTER_NEAREST)
            if box:
                x0, y0, x1, y1 = (value * CHECK_ZOOM for value in box)
                cv2.rectangle(check, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
            cv2.putText(check, symbol or "nothing", (3, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
            folder = self.root / CHECK_DIR
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"{split}_{name}.png"), check)


def even_split(total: int, capacities: Dict[str, Optional[int]], rng: random.Random) -> Dict[str, int]:
    """
    total split between the keys as evenly as possible: the same share for each key, rounded (the remainder,
    one more each, to keys drawn at random), never more than its capacity (None: no limit). A key that can't
    take its share takes all it can and the rest is split between the others. The sum of the shares is total,
    or the sum of the capacities if it is less.
    """
    room = {key: math.inf if capacity is None else capacity for key, capacity in capacities.items()}
    shares: Dict[str, int] = {}
    left = total
    remaining = sorted(capacities, key=lambda key: (room[key], key))  # Smallest capacity first
    # The keys that can't take the even share of what is left take all they can (the share of the others grows).
    while remaining and room[remaining[0]] <= left // len(remaining):
        key = remaining.pop(0)
        shares[key] = room[key]
        left -= shares[key]
    if remaining:  # Each one can take one more than the even share: the remainder to keys drawn at random
        each, extra = divmod(left, len(remaining))
        lucky = set(rng.sample(sorted(remaining), extra))
        shares.update({key: each + (key in lucky) for key in remaining})
    return {key: shares[key] for key in capacities}


def balanced_counts(total: int, classes: List[str], available: Optional[Dict[str, int]],
                    rng: random.Random) -> Dict[str, int]:
    """
    The number of images of each class, total in all: the same number for every set (the icon of the class:
    'atq' as much as 'mrd'), then, inside a set, the same number for each of its classes (its rarities),
    rounded (see even_split). available: the most images of each class (the real cards of the validation),
    None: no limit (generated images).
    """
    by_set: Dict[str, List[str]] = defaultdict(list)
    for label in classes:
        by_set[parse_class(label)[0]].append(label)

    def capacity(label: str) -> Optional[int]:
        return None if available is None else available.get(label, 0)

    per_set = even_split(total, {icon: None if available is None else sum(capacity(label) for label in labels)
                                 for icon, labels in by_set.items()}, rng)
    counts: Dict[str, int] = {}
    for icon in sorted(by_set):
        counts.update(even_split(per_set[icon], {label: capacity(label) for label in by_set[icon]}, rng))
    return counts


def per_set_counts(per_class: Dict[str, int]) -> Dict[str, int]:
    """The images of each set (the icon of the class), from the images of each class."""
    counts: Dict[str, int] = defaultdict(int)
    for label, count in per_class.items():
        counts[parse_class(label)[0]] += count
    return dict(sorted(counts.items()))


def _print_report(counts: Dict[str, Counter], classes: List[str], extra: dict, root: Path):
    report = {}
    for split in ("train", "val"):
        split_counts = counts.get(split, Counter())
        per_class = {name: split_counts[name] for name in classes}
        per_set = per_set_counts(per_class)
        present = [value for value in per_class.values() if value]
        present_sets = [value for value in per_set.values() if value]
        report[split] = {"images": split_counts["_images"], "nothing": split_counts["_nothing"],
                         "per_set": per_set, "per_class": per_class}
        print(f"  {split}: {split_counts['_images']} images ({split_counts['_nothing']} without symbol), "
              f"{len(present_sets)}/{len(per_set)} sets"
              + (f", {min(present_sets)} to {max(present_sets)} per set" if present_sets else "")
              + f", {len(present)}/{len(classes)} classes"
              + (f", {min(present)} to {max(present)} per class" if present else ""))
    report.update(extra)
    with open(root / "report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=4, ensure_ascii=False)


def _pick_val_cards(cards: List[CardImage], classes: List[str], size: Sizes, rng: random.Random
                    ) -> Tuple[List[CardImage], List[CardImage]]:
    """
    The validation cards: size.val_nothing old cards (no symbol), then the cards with a symbol, the same
    number for every set, split evenly between its rarities (balanced_counts, within the real cards on disk),
    in random order.
    :return: (validation cards, old cards left for the training: not a card name of the validation).
    """
    old = sorted((card for card in cards if card.old), key=lambda card: str(card.path))
    rng.shuffle(old)
    old_val = old[:size.val_nothing]
    if len(old_val) < size.val_nothing:
        raise ValueError(f"{size.val_nothing} validation cards without symbol wanted, only {len(old)} old cards "
                         f"(before Arabian Nights: lea, leb, 2ed) on disk: download them or lower the percent.")
    val_names = {card.card.get("name") for card in old_val}
    old_left = [card for card in old if card.card.get("name") not in val_names]
    by_class: Dict[str, List[CardImage]] = defaultdict(list)
    wanted = set(classes)
    for card in cards:
        if card.label in wanted and card.for_val:
            by_class[card.label].append(card)
    for label in sorted(by_class):
        rng.shuffle(by_class[label])
    counts = balanced_counts(size.val - size.val_nothing, classes,
                             {label: len(by_class[label]) for label in classes}, rng)
    picked = [card for label in sorted(by_class) for card in by_class[label][:counts[label]]]
    rng.shuffle(picked)
    _print_val_shortages(counts, classes, {label: len(by_class[label]) for label in classes})
    return old_val + picked, old_left


def _short(counts: Dict[str, int], available: Dict[str, int]) -> Tuple[List[str], int]:
    """
    The keys of counts that got all their available cards and fewer than the others: (keys, the least the
    others got).
    """
    others = [count for key, count in counts.items() if count < available[key]]
    level = min(others) if others else max(counts.values(), default=0)
    return sorted((key for key, count in counts.items() if count < level), key=lambda key: (counts[key], key)), level


def _print_val_shortages(counts: Dict[str, int], classes: List[str], available: Dict[str, int]):
    """The sets (and the rarities inside a set) with too few real cards to get their share of the validation."""
    def listed(keys: List[str], number: Dict[str, int]) -> str:
        return ", ".join(f"{key} {number[key]}" for key in keys[:12]) + ("..." if len(keys) > 12 else "")

    per_set, available_per_set = per_set_counts(counts), per_set_counts(available)
    short_sets, level = _short(per_set, available_per_set)
    if short_sets:
        print(f"Val: {len(short_sets)} sets with too few real cards for their share ({level} or {level + 1}), all "
              f"their cards taken, the rest given to the other sets: {listed(short_sets, per_set)}")
    short_classes = []
    for icon in per_set:
        labels = [label for label in classes if parse_class(label)[0] == icon]
        short_classes += _short({label: counts[label] for label in labels}, available)[0]
    if short_classes:
        print(f"Val: {len(short_classes)} rarities with too few real cards for their share in their set, all their "
              f"cards taken, the rest given to the other rarities of the set: {listed(short_classes, counts)}")


def build_dataset(n: int, val_percent: float, no_symbol_percent: float, classes: List[str],
                  cards: List[CardImage], stamp: dict, val_augmentation: Optional[dict] = None
                  ) -> Tuple[Path, List[str]]:
    """
    Write dataset_dir(...) (removed first), dataset.stamp last. val_augmentation: see val_augmentation_of
    (None: the real cards as they are). :return: (data.yaml path, classes).
    """
    root = dataset_dir(n, val_percent, no_symbol_percent)
    if root.exists():
        shutil.rmtree(root)
    rng = random.Random(SEED)
    size = sizes(n, val_percent, no_symbol_percent)
    print(f"Set symbol dataset of {n} samples: {size.train} generated for train ({size.train_nothing} without "
          f"symbol), {size.val} real cards for val ({size.val_nothing} without symbol): {root}")
    symbols = {}
    for label in tqdm.tqdm(classes, desc="Read the symbols", unit="symbol"):
        symbols[label] = load_symbol(symbol_file(label))
    class_to_index = {name: index for index, name in enumerate(classes)}
    writer = _Writer(root, class_to_index)
    print(f"{len(classes)} classes, {len(cards)} real card images ({sum(card.old for card in cards)} old cards, "
          f"before Arabian Nights).")

    # Val first: real cards, without box. They are then kept out of the training.
    val_cards, old_left = _pick_val_cards(cards, classes, size, rng)
    frames = Counter()
    augment = _ValAugmenter(val_augmentation) if val_augmentation else None
    for card in tqdm.tqdm(val_cards, desc="Real cards val", unit="image"):
        card_bgr = read_card(card.path)
        if card_bgr is None:
            continue
        window = real_window(card_bgr)
        if augment is not None:
            window = augment(window)
        writer.write("val", window, card.label, None, str(card.path))
        frames["old" if card.old else card.frame] += 1
    if len(val_cards) < size.val:
        print(f"Only {len(val_cards)} real card images for the validation ({size.val} wanted): download more cards.")
    if not val_cards:
        raise ValueError("No real card image for the validation.")
    print(f"Val frames: {dict(frames)}")

    # The empty symbol zones of the old cards left (the images without symbol, the "old_symbol_zone" backgrounds).
    old_zones = []
    for card in tqdm.tqdm(old_left, desc="Old symbol zones", unit="image"):
        card_bgr = read_card(card.path)
        if card_bgr is not None:
            old_zones.append((real_window(card_bgr), str(card.path)))
    if not old_zones and (size.train_nothing or BACKGROUNDS.get("old_symbol_zone")):
        raise ValueError("No old card (before Arabian Nights: lea, leb, 2ed) left for the training: download them.")

    # Train: generated images.
    val_paths = {card.path for card in val_cards}
    pool = sorted({card.path for card in cards if card.path not in val_paths})
    if not pool:
        raise FileNotFoundError(f"No card image for the backgrounds in {local_paths.EXP_DIR}.")
    names = sorted({card.card["name"] for card in cards if card.card.get("name")})
    fonts = text_fonts()
    print(f"Backgrounds: {len(pool)} cards, {len(old_zones)} old symbol zones. "
          f"Text fonts: {', '.join(font.name for font in fonts) or 'OpenCV Hershey'}.")
    # The class of each image (None: no symbol): the same number for every set, then for each of its rarities.
    counts = balanced_counts(size.train - size.train_nothing, classes, None, rng)
    labels = [None] * size.train_nothing + [label for label in classes for _ in range(counts[label])]
    rng.shuffle(labels)
    kinds, weights = zip(*BACKGROUNDS.items())
    used = Counter()
    for label in tqdm.tqdm(labels, desc="Generate train", unit="image"):
        if label is None:
            zone, path = rng.choice(old_zones)
            writer.write("train", zone, None, None, f"old_symbol_zone:{path}")
            used["without symbol"] += 1
            continue
        kind = rng.choices(kinds, weights)[0]
        if kind == "old_symbol_zone":
            zone, path = rng.choice(old_zones)
            background, source = zone, f"{kind}:{path}"
        elif kind == "color":
            background, source = color_background(rng), "color"
        elif kind == "gradient":
            background, source = gradient_background(rng), "gradient"
        else:
            card_bgr = None
            while card_bgr is None:
                path = rng.choice(pool)
                card_bgr = read_card(path)
            background, source = card_background(card_bgr, kind, rng), f"{kind}:{path}"
        used[kind] += 1
        if kind != "old_symbol_zone" and rng.random() < TEXT_RATIO:
            background = draw_text(background, names, fonts, rng)
            source += " + text"
            used["text"] += 1
        layer, alpha = symbols[label]
        image, box = paste_symbol(background, layer, alpha, rng)
        writer.write("train", image, label, box, f"{source} + {symbol_file(label).name}")
    print(f"Train images: {dict(used)}")

    with open(root / "data.yaml", "w", encoding="utf-8") as fp:
        fp.write(f"path: {json.dumps(root.resolve().as_posix())}\n"
                 f"train: images/train\n"
                 f"val: images/val\n"
                 f"nc: {len(classes)}\n"
                 f"names:\n" + "".join(f"  {i}: {json.dumps(name)}\n" for i, name in enumerate(classes)))
    with open(root / training.CLASSES_FILE, "w", encoding="utf-8") as fp:
        json.dump(classes, fp, indent=4, ensure_ascii=False)
    with open(root / "images.json", "w", encoding="utf-8") as fp:
        json.dump(writer.entries, fp, indent=4, ensure_ascii=False)
    _print_report(writer.counts, classes, {"val_frames": dict(frames), "train_used": dict(used)}, root)
    training._write_stamp(root / STAMP_FILE, stamp)
    print(f"Set symbol dataset written: {root} (check images in {root / CHECK_DIR})")
    return root / "data.yaml", classes


def check_sizes(n: int, val_percent: float, no_symbol_percent: float = 0.0) -> Optional[str]:
    """Why these settings can't make a dataset, None if they can."""
    if n < MIN_SAMPLES:
        return f"at least {MIN_SAMPLES} samples"
    if not 0 < val_percent < 100:
        return "a validation percent between 0 and 100"
    if not 0 <= no_symbol_percent < 100:
        return "a percent without symbol from 0 to less than 100"
    size = sizes(n, val_percent, no_symbol_percent)
    if size.val < 1:
        return f"at least one validation image: {val_percent:g}% of {n} is less than 1"
    if size.train < 1:
        return "at least one generated image for the training"
    if size.val - size.val_nothing < 1 or size.train - size.train_nothing < 1:
        return "at least one image with a symbol in each split"
    return None


def load_or_build_dataset(n: int, val_percent: float = VAL_PERCENT, no_symbol_percent: float = NO_SYMBOL_PERCENT,
                          force: bool = False, val_augmentation: Optional[dict] = None) -> Tuple[Path, List[str]]:
    """
    The dataset of n samples, val_percent% of them real cards for the validation, no_symbol_percent% of
    each split without symbol, built unless its dataset.stamp matches (force: always). val_augmentation:
    the augmentation of the real cards of the validation (val_augmentation_of), None: none.
    :return: (data.yaml path, classes).
    """
    problem = check_sizes(n, val_percent, no_symbol_percent)
    if problem:
        raise ValueError(f"{n} samples, {val_percent:g}% of validation, {no_symbol_percent:g}% without symbol: {problem}")
    root = dataset_dir(n, val_percent, no_symbol_percent)
    classes_path = root / training.CLASSES_FILE
    if not force and (root / "data.yaml").exists() and classes_path.exists() and (root / STAMP_FILE).exists():
        with open(classes_path, "r", encoding="utf-8") as fp:
            classes = json.load(fp)
        if all(symbol_file(name).exists() for name in classes) and \
                training._stamp_matches(root / STAMP_FILE, dataset_stamp(n, val_percent, no_symbol_percent, classes,
                                                                         val_augmentation)):
            print(f"Set symbol dataset loaded from {root} ({STAMP_FILE} is up to date).")
            return root / "data.yaml", classes
    classes, cards, _ = classes_and_cards()
    if not classes:
        raise ValueError(f"No set symbol with real cards: download the set symbols (download.py, choice 3, which "
                         f"also colors them) and cards. Colored SVG files expected in {symbol_dir()}.")
    print(f"{len(classes)} classes: {sum(parse_class(name)[1] is None for name in classes)} sets before Exodus, "
          f"{Counter(parse_class(name)[1] for name in classes if parse_class(name)[1])}")
    return build_dataset(n, val_percent, no_symbol_percent, classes, cards,
                         dataset_stamp(n, val_percent, no_symbol_percent, classes, val_augmentation),
                         val_augmentation)


def _ask_percent(question: str, default: float) -> Optional[float]:
    """A percent typed by the user (Enter: default, '2,5' and '2.5%' accepted), None if it is not a number."""
    answer = input(f"{question} [{default:g}] ? : ").strip()
    try:
        return float(answer.replace(",", ".").rstrip("%")) if answer else default
    except ValueError:
        print("Answer a number, ex: 2.5")
        return None


def ask_samples() -> int:
    """Number of samples n, asked to the user."""
    while True:
        answer = input(f"How many samples (n, at least {MIN_SAMPLES}) ? : ").strip()
        if answer.isdigit() and int(answer) >= MIN_SAMPLES:
            return int(answer)
        print(f"Answer a number of at least {MIN_SAMPLES}.")


def ask_val_percent(n: int) -> float:
    """Percent of the n samples for the validation (real cards), asked to the user (Enter: VAL_PERCENT)."""
    while True:
        percent = _ask_percent("Percent of the samples for the validation (real cards)", VAL_PERCENT)
        if percent is None:
            continue
        problem = check_sizes(n, percent)
        if problem is None:
            size = sizes(n, percent, 0)
            print(f"Validation: {size.val} real cards, training: {size.train} generated images.")
            return percent
        print(f"Not possible: {problem}.")


def ask_no_symbol_percent(n: int, val_percent: float) -> float:
    """Percent of each split without symbol (old cards), asked to the user (Enter: NO_SYMBOL_PERCENT)."""
    while True:
        percent = _ask_percent("Percent of the images without symbol (old cards, train and val)", NO_SYMBOL_PERCENT)
        if percent is None:
            continue
        problem = check_sizes(n, val_percent, percent)
        if problem is None:
            size = sizes(n, val_percent, percent)
            print(f"Without symbol: {size.val_nothing} of the validation, {size.train_nothing} of the training.")
            return percent
        print(f"Not possible: {problem}.")


def main():
    parser = argparse.ArgumentParser(description="Dataset of the set symbol detector (see the top of the file).")
    parser.add_argument("n", nargs="?", type=int, help="number of samples (asked if not given)")
    parser.add_argument("--val", type=float, help="percent of the samples for the validation (asked if not given)")
    parser.add_argument("--nosym", type=float, help="percent of each split without symbol (asked if not given)")
    parser.add_argument("--force", action="store_true", help="build it even if its dataset.stamp is up to date")
    args = parser.parse_args()
    n = args.n if args.n else ask_samples()
    val_percent = args.val if args.val is not None else ask_val_percent(n)
    no_symbol_percent = args.nosym if args.nosym is not None else ask_no_symbol_percent(n, val_percent)
    import training_set_symbol  # The augmentation of the set_gen training, also given to the validation
    load_or_build_dataset(n, val_percent, no_symbol_percent, args.force,
                          training_set_symbol.val_augmentation(training_set_symbol.definitions()[0]))


if __name__ == "__main__":
    main()
