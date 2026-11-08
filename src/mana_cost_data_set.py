"""
Dataset of the mana cost detector: generated images to train, real cards to validate.

    python3 src/mana_cost_data_set.py [n] [--force]     (n is asked if not given)

Also built by the 'mana_cost_gen' training of training_mana_cost.py, which trains on it.

n samples, in <runs_dir>/detect/mana_cost/gen/<n>/ (in WSL: ~/mtg/runs, see local_paths.py):
- train: n - n_val generated images of IMAGE_WIDTH x IMAGE_HEIGHT (300 x 110):
  1. Background (BACKGROUNDS), 25% each:
     - "art": a random card image of the 2015 frame (on disk), resized to CARD_WIDTH x CARD_HEIGHT
       (488 x 680), the 300 x 110 box in the middle of its top half (ART_BOX, in the art),
     - "top_left": the 300 x 110 box at the top left of such a card (TOP_LEFT_BOX: border, name bar,
       start of the name), only on cards whose mana cost stays out of that box,
     - "color": one random color,
     - "gradient": a gradient between two random colors, in a random direction.
     The cards of the validation are never used as backgrounds.
  2. Text, on TEXT_RATIO (15%) of the images: a few lines of card names, each line black or white,
     at random places (no box: the text is not a symbol).
  3. A random mana cost built like the ones of the cards (see random_mana_cost): each part of
     MANA_COST_PARTS is in it with its probability ({X}/{Y}/{Z} 5%, a numbered mana 50%, {2/W}... 2%,
     {C} 2%, {C/W}... 1%, colored mana 80%, {W/U}... 4%, {W/P}... 3%, {W/U/P}... 1%, {S} 2%), with
     SYMBOL_REPEATS symbols (one for the numbered mana), and the symbols are sorted in CARD_ORDER.
     The symbols come from scryfall/symbol/ (download them first: python3 src/download.py, choice 2).
     A new mana cost is picked while it is empty or wider than the image.
  4. The symbols in one row, SYMBOL_GAP (2) pixels apart, their centers at the same height. A
     symbol is 23 x 23 pixels (measured on real cards: ~23 px wide, centers ~25.5 px apart), a hybrid
     one (a '-' in its file name: R-G.svg, 2-W.svg, U-P.svg) is 30 x 30. The row is put at a random
     place, fully inside the image, over the text. On SHADOW_PROBABILITY (50%) of the images, every
     symbol has a small dark shadow below on the left, as printed on the cards (not in its box).
  5. One box per symbol, its class = the text between the braces in a mana cost: W-U.svg -> 'W/U'
     (same classes as training_mana_cost.py).
- val: n_val = round(n * VAL_RATIO), at most VAL_MAX (20%, at most 4000) real cards: every sample
  beyond goes to train. The 300 x 110 window at the top right of the card (VAL_RECT,
  also the search window of the detector). Random card images on disk, of the frames, layouts and
  filters of training_mana_cost.py (a land = "nothing"). Only the mana cost of the card is known, so
  its boxes are a simple guess (see real_window): what counts is the mana cost read, not the boxes.
  Skipped: the cards with a symbol that is not a class, and the mana costs going out of the window.

Files: data.yaml, classes.json, images/{train,val}/*.png + labels/{train,val}/*.txt (YOLO format),
images.json (content of each image), report.json (boxes per class), check/ (first images of each
split with their boxes), and dataset.stamp: the settings of the dataset, written last. The dataset is
not built again while dataset.stamp matches (--force: always). New card images downloaded later
don't change the stamp: use --force to use them.
"""
import argparse
import functools
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image, ImageDraw, ImageFont

import download
import local_paths
import training
import training_mana_cost as mana

VERSION = 5   # Change it when the generation changes: every dataset is built again
SEED = 0      # Same n -> same dataset

# --- Symbols ---
SYMBOL_DIR = local_paths.SCRYFALL_DIR / download.SYMBOL_DIR_NAME
# Symbol files never used (name without .svg, any case).
EXCLUDED = ["100", "1000000", "CHAOS", "D", "E", "H", "HR", "HW", "L", "P", "PW", "Q", "T", "TK"]
SYMBOL_SIZE = 23        # Width and height of a symbol, pixels (real cards: ~23 px)
HYBRID_SIZE = 30        # Width and height of a hybrid symbol (a '-' in its file name)
SYMBOL_GAP = 2          # Pixels between two symbols of a mana cost (real cards: ~2.5 px)
# Dark shadow under the symbols, as printed on the cards: on SHADOW_PROBABILITY of the generated images,
# every symbol gets one, its own shape moved by SHADOW_OFFSET (x, y) pixels (left and down), softened
# by SHADOW_BLUR (Gaussian sigma, pixels), SHADOW_OPACITY black. Drawn under all the symbols.
SHADOW_PROBABILITY = 0.5
SHADOW_OFFSET = (-2, 2)
SHADOW_BLUR = 0.8
SHADOW_OPACITY = 0.8
_SHADOW_PAD = 4         # Margin around the shadow of a symbol, for its offset and its blur
# --- Mana cost of a generated image, built like on the cards (see random_mana_cost) ---
# The parts of a mana cost, in the order of the cards: (part, probability that a mana cost has it, its
# symbols). Each part is drawn on its own.
MANA_COST_PARTS = [
    ("variable", 0.05, ["X", "Y", "Z"]),
    ("number", 0.50, []),  # One symbol, drawn with number_weights()
    ("two_hybrid", 0.02, ["2/W", "2/U", "2/B", "2/R", "2/G"]),
    ("colorless", 0.02, ["C"]),
    ("colorless_hybrid", 0.01, ["C/W", "C/U", "C/B", "C/R", "C/G", "C/P"]),
    ("colored", 0.80, ["W", "U", "B", "R", "G"]),
    ("hybrid", 0.04, ["W/U", "W/B", "U/B", "U/R", "B/R", "B/G", "R/G", "R/W", "G/W", "G/U"]),
    ("phyrexian", 0.03, ["W/P", "U/P", "B/P", "R/P", "G/P"]),
    ("hybrid_phyrexian", 0.01, ["W/U/P", "W/B/P", "U/B/P", "U/R/P", "B/R/P", "B/G/P", "R/G/P", "R/W/P",
                                "G/W/P", "G/U/P"]),
    ("snow", 0.02, ["S"]),
]
# Numbered mana: {1} to {7} weighted as on the cards, sharing what RARE_NUMBERS leave (RARE_NUMBER_SHARE each).
NUMBER_WEIGHTS = {"1": 25, "2": 25, "3": 18, "4": 12, "5": 7, "6": 5, "7": 3}
RARE_NUMBERS = ["0", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "½", "∞", "A"]
RARE_NUMBER_SHARE = 0.01
# Number of symbols of a part that is in the mana cost (not the numbered mana: always one): {count: probability}.
SYMBOL_REPEATS = {1: 0.50, 2: 0.30, 3: 0.10, 4: 0.05, 5: 0.025, 6: 0.025}
# Number of different symbols among them, as on the cards ({U}{U}, {W}{U}{U}, {W}{U}{B}): {count: probability}.
DISTINCT_SYMBOLS = {1: 0.70, 2: 0.25, 3: 0.05}
# Order of the symbols in a mana cost: variables, numbered mana, {2/W}..., colorless, colored mana with their
# hybrids between them, Phyrexian, hybrid Phyrexian, snow.
CARD_ORDER = (["X", "Y", "Z"] + list(NUMBER_WEIGHTS) + RARE_NUMBERS + ["2/W", "2/U", "2/B", "2/R", "2/G"]
              + ["C", "C/W", "C/U", "C/B", "C/R", "C/G", "C/P"]
              + ["W", "W/U", "W/B", "U", "U/B", "U/R", "B", "B/R", "B/G", "R", "R/G", "R/W", "G", "G/W", "G/U"]
              + ["W/P", "U/P", "B/P", "R/P", "G/P"]
              + ["W/U/P", "W/B/P", "U/B/P", "U/R/P", "B/R/P", "B/G/P", "R/G/P", "R/W/P", "G/W/P", "G/U/P"]
              + ["S"])
RENDER_SCALE = 4        # The SVG is drawn 4x bigger, then reduced: smooth edges

# --- Images ---
IMAGE_WIDTH, IMAGE_HEIGHT = 300, 110
VAL_RATIO = 0.2
VAL_MAX = 4000          # At most this many real cards for the validation, whatever n
MIN_SAMPLES = 5         # At least one validation image
# Background of a generated image: {kind: probability} (see the top of the file).
BACKGROUNDS = {"art": 0.25, "top_left": 0.25, "color": 0.25, "gradient": 0.25}
BACKGROUND_FRAME = "2015"   # Cards of the "art" and "top_left" backgrounds
# In a CARD_WIDTH x CARD_HEIGHT card: centered in the top half (the art), and the top left corner.
_ART_X = (training.CARD_WIDTH - IMAGE_WIDTH) // 2
_ART_Y = training.CARD_HEIGHT // 4 - IMAGE_HEIGHT // 2
ART_BOX = (_ART_X, _ART_Y, _ART_X + IMAGE_WIDTH, _ART_Y + IMAGE_HEIGHT)
TOP_LEFT_BOX = (0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)
TOP_LEFT_MARGIN = 5         # Pixels between TOP_LEFT_BOX and the mana cost of the card (as real_window guesses it)
# Text drawn on TEXT_RATIO of the generated images: TEXT_LINES lines of card names, font size TEXT_SIZES
# pixels, each line black or white. Fonts: every TEXT_FONT_FILES found in TEXT_FONT_DIRS (Windows fonts
# seen from WSL, then Linux ones), else the OpenCV Hershey fonts.
TEXT_RATIO = 0.15
TEXT_LINES = (1, 4)
TEXT_SIZES = (12, 26)
TEXT_FONT_DIRS = ["/mnt/c/Windows/Fonts", "C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu"]
TEXT_FONT_FILES = ["timesbd.ttf", "times.ttf", "georgiab.ttf", "georgia.ttf", "palab.ttf", "cambriab.ttf",
                   "arialbd.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSans-Bold.ttf"]
HERSHEY_FONTS = [cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]
# Window of the real cards (validation) and search window of the detector: top right of the card.
VAL_RECT = training.Rect(center_x=training.CARD_WIDTH - IMAGE_WIDTH / 2, center_y=IMAGE_HEIGHT / 2,
                         height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
# Boxes of the real cards: the mana cost row, VAL_BOX_TOP pixels from the top of the card and
# VAL_BOX_RIGHT from its right, on every card (see real_window).
VAL_BOX_TOP = 37
VAL_BOX_RIGHT = 37

# --- Files ---
OUTPUT_DIR = training.DETECT_OUTPUT_DIR / "mana_cost" / "gen"
STAMP_FILE = "dataset.stamp"
CHECK_DIR = "check"
CHECK_IMAGES = 50       # First images of each split saved with their boxes in check/
CHECK_ZOOM = 2

Box = training.Box


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    name: str           # Class: 'W', 'W/U', '2/W'...
    size: int           # Width = height, pixels
    image: np.ndarray   # size x size, premultiplied BGRA float32: (B * A, G * A, R * A, A), A in 0-1
    shadow: np.ndarray  # Alpha (0-1) of its shadow, (size + 2 * _SHADOW_PAD) x (size + 2 * _SHADOW_PAD), not moved


def symbol_class(path: Path) -> str:
    """W.svg -> 'W', W-U.svg -> 'W/U', G-W-P.svg -> 'G/W/P' (the text between the braces)."""
    return path.stem.replace("-", "/")


def symbol_size(path: Path) -> int:
    """HYBRID_SIZE for a hybrid symbol (a '-' in its file name), else SYMBOL_SIZE."""
    return HYBRID_SIZE if "-" in path.stem else SYMBOL_SIZE


def symbol_files(symbol_dir: Path = SYMBOL_DIR) -> List[Path]:
    """The .svg files of symbol_dir, without EXCLUDED, sorted."""
    excluded = {name.lower() for name in EXCLUDED}
    paths = sorted(path for path in symbol_dir.glob("*.svg") if path.stem.lower() not in excluded)
    if not paths:
        raise FileNotFoundError(f"No symbol in {symbol_dir}: download them first (python3 src/download.py, choice 2).")
    return paths


def render_svg(path: Path, size: int) -> np.ndarray:
    """An SVG file as a size x size premultiplied BGRA float32 image (see Symbol)."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # PyMuPDF before 1.24
        except ImportError as error:
            raise ImportError(f"{path.name}: the SVG symbols need PyMuPDF: pip install pymupdf") from error
    big = size * RENDER_SCALE
    with pymupdf.open(str(path)) as document:
        page = document[0]
        matrix = pymupdf.Matrix(big / page.rect.width, big / page.rect.height)
        png = page.get_pixmap(matrix=matrix, alpha=True).tobytes("png")
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Can't draw {path} with transparency")
    image = image.astype(np.float32)
    alpha = image[:, :, 3:4] / 255
    # Premultiplied, so that reducing it doesn't darken its edges with the color of the transparent pixels.
    premultiplied = np.concatenate([image[:, :, :3] * alpha, alpha], axis=2)
    return cv2.resize(premultiplied, (size, size), interpolation=cv2.INTER_AREA)


def shadow_alpha(image: np.ndarray) -> np.ndarray:
    """Alpha of the shadow of a symbol image (see Symbol): its shape, softened, SHADOW_OPACITY dark, padded."""
    alpha = np.pad(image[:, :, 3], _SHADOW_PAD)
    return cv2.GaussianBlur(alpha, (0, 0), SHADOW_BLUR) * SHADOW_OPACITY


def load_symbols(paths: List[Path]) -> List[Symbol]:
    symbols = []
    for path in tqdm.tqdm(paths, desc="Draw symbols", unit="symbol"):
        image = render_svg(path, symbol_size(path))
        symbols.append(Symbol(symbol_class(path), symbol_size(path), image, shadow_alpha(image)))
    return symbols


def row_width(sizes: List[int], gap: int = SYMBOL_GAP) -> int:
    """Width of a row of symbols of these sizes, `gap` pixels apart."""
    return sum(sizes) + gap * max(0, len(sizes) - 1)


def number_weights() -> Dict[str, float]:
    """Probability of each numbered mana: RARE_NUMBER_SHARE for each RARE_NUMBERS, the rest as NUMBER_WEIGHTS."""
    common = 1 - RARE_NUMBER_SHARE * len(RARE_NUMBERS)
    total = sum(NUMBER_WEIGHTS.values())
    return {**{name: common * weight / total for name, weight in NUMBER_WEIGHTS.items()},
            **{name: RARE_NUMBER_SHARE for name in RARE_NUMBERS}}


def _draw(weights: Dict, rng: random.Random):
    values, probabilities = zip(*weights.items())
    return rng.choices(values, probabilities)[0]


def _part_symbols(part_symbols: List[str], rng: random.Random) -> List[str]:
    """SYMBOL_REPEATS symbols of a part, spread over DISTINCT_SYMBOLS different ones: {U}{U}, {W}{U}{U}..."""
    count = _draw(SYMBOL_REPEATS, rng)
    distinct = rng.sample(part_symbols, min(_draw(DISTINCT_SYMBOLS, rng), count, len(part_symbols)))
    return distinct + [rng.choice(distinct) for _ in range(count - len(distinct))]


def check_mana_cost_symbols(symbols: List[Symbol]):
    """Every symbol of CARD_ORDER must be a class, and every class in CARD_ORDER (else it is never generated)."""
    names = {symbol.name for symbol in symbols}
    missing = [name for name in CARD_ORDER if name not in names]
    if missing:
        raise ValueError(f"Symbols of CARD_ORDER missing in {SYMBOL_DIR} (or EXCLUDED): {missing}")
    never = sorted(names - set(CARD_ORDER))
    if never:
        print(f"Warning: never generated (not in CARD_ORDER): {never}")


def random_mana_cost(symbols: List[Symbol], rng: random.Random) -> List[Symbol]:
    """
    A mana cost built like on the cards: each part of MANA_COST_PARTS is in it with its probability, the
    numbered mana with one symbol (number_weights), the other parts with SYMBOL_REPEATS symbols, then
    the symbols are sorted in CARD_ORDER: {X}{2}{W}{W/U}{U}. Picked again while it is empty or wider
    than the image.
    """
    by_name = {symbol.name: symbol for symbol in symbols}
    numbers = number_weights()
    while True:
        names = []
        for part, probability, part_symbols in MANA_COST_PARTS:
            if rng.random() < probability:
                names += [_draw(numbers, rng)] if part == "number" else _part_symbols(part_symbols, rng)
        if not names:
            continue
        names.sort(key=CARD_ORDER.index)
        row = [by_name[name] for name in names]
        if row_width([symbol.size for symbol in row]) <= IMAGE_WIDTH:
            return row


def _darken(image: np.ndarray, alpha: np.ndarray, x0: int, y0: int):
    """Darken `image` (float32) by `alpha` (0-1) placed at (x0, y0), cut at the image edges."""
    height, width = alpha.shape
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(image.shape[1], x0 + width), min(image.shape[0], y0 + height)
    if ix1 > ix0 and iy1 > iy0:
        image[iy0:iy1, ix0:ix1] *= 1 - alpha[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0, None]


def draw_mana_cost(background_bgr: np.ndarray, row: List[Symbol], rng: random.Random,
                   shadow_probability: float = SHADOW_PROBABILITY, gap: int = SYMBOL_GAP
                   ) -> Tuple[np.ndarray, List[Tuple[str, Box]]]:
    """
    The row of symbols at a random place of the background (fully inside): `gap` pixels apart, their
    centers on the same line (half a pixel off between symbols of an odd size difference). With the
    probability `shadow_probability`, every symbol has its dark shadow (SHADOW_OFFSET...), drawn under
    all the symbols. The boxes are the symbols only, not their shadow.
    :return: the image and (class, box) of each symbol, from left to right.
    """
    if not row:
        return background_bgr.copy(), []
    row_height = max(symbol.size for symbol in row)
    x = rng.randint(0, IMAGE_WIDTH - row_width([symbol.size for symbol in row], gap))
    top = rng.randint(0, IMAGE_HEIGHT - row_height)
    places = []
    for symbol in row:
        places.append((symbol, x, top + (row_height - symbol.size) // 2))
        x += symbol.size + gap
    image = background_bgr.astype(np.float32)
    if rng.random() < shadow_probability:
        for symbol, sx, sy in places:
            _darken(image, symbol.shadow, sx - _SHADOW_PAD + SHADOW_OFFSET[0], sy - _SHADOW_PAD + SHADOW_OFFSET[1])
    boxes = []
    for symbol, sx, sy in places:
        size = symbol.size
        below = image[sy:sy + size, sx:sx + size]
        image[sy:sy + size, sx:sx + size] = symbol.image[:, :, :3] + below * (1 - symbol.image[:, :, 3:4])
        boxes.append((symbol.name, (sx, sy, sx + size, sy + size)))
    return np.clip(image + 0.5, 0, 255).astype(np.uint8), boxes


# ---------------------------------------------------------------------------
# Backgrounds and text
# ---------------------------------------------------------------------------

def _random_color(rng: random.Random) -> np.ndarray:
    return np.array([rng.randint(0, 255) for _ in range(3)], np.float32)


def color_background(rng: random.Random) -> np.ndarray:
    """One random color."""
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), _random_color(rng), np.uint8)


def gradient_background(rng: random.Random) -> np.ndarray:
    """A linear gradient between two random colors, in a random direction."""
    angle = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH].astype(np.float32)
    position = xs * np.cos(angle) + ys * np.sin(angle)
    position = ((position - position.min()) / (position.max() - position.min()))[:, :, None]
    first, second = _random_color(rng), _random_color(rng)
    return np.clip(first * (1 - position) + second * position + 0.5, 0, 255).astype(np.uint8)


def card_background(path: Path, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """`box` of a card image resized to CARD_WIDTH x CARD_HEIGHT, None if unreadable."""
    card_bgr = read_card(path)
    if card_bgr is None:
        return None
    x0, y0, x1, y1 = box
    return card_bgr[y0:y1, x0:x1].copy()


def make_background(kind: str, pools: Dict[str, List[Path]], rng: random.Random) -> Tuple[np.ndarray, str]:
    """A background of that kind (see BACKGROUNDS). :return: (image, its source for images.json)."""
    if kind == "color":
        return color_background(rng), "color"
    if kind == "gradient":
        return gradient_background(rng), "gradient"
    while True:
        path = rng.choice(pools[kind])
        image = card_background(path, ART_BOX if kind == "art" else TOP_LEFT_BOX)
        if image is not None:
            return image, f"{kind}:{path}"


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
        x = rng.randint(-IMAGE_WIDTH // 4, IMAGE_WIDTH - 2 * size)
        y = rng.randint(-size // 3, IMAGE_HEIGHT - size)
        color = (0, 0, 0) if rng.random() < 0.5 else (255, 255, 255)
        lines.append((text, size, x, y, color))
    if not fonts:
        image = image_bgr.copy()
        for text, size, x, y, color in lines:
            font = rng.choice(HERSHEY_FONTS)
            scale = 0.7 * size / cv2.getTextSize("H", font, 1.0, 1)[0][1]
            cv2.putText(image, text, (x, y + size), font, scale, color, rng.choice([1, 2]), cv2.LINE_AA)
        return image
    image = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]))  # Pillow: RGB
    draw = ImageDraw.Draw(image)
    for text, size, x, y, color in lines:
        draw.text((x, y), text, fill=color, font=_font(rng.choice(fonts), size))
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

@dataclass
class CardImage:
    path: Path
    exp_code: str
    card: dict          # The card of info.json (frame, set_type...)
    mana_cost: str      # Of the face: '{1}{W}{W}', '' = no mana cost (land, back face...)


def list_card_images() -> List[CardImage]:
    """
    The card images on disk of the cards accepted by the mana cost training (training_mana_cost.py:
    FRAMES, LAYOUTS, FILTERS), one per illustration and mana cost.
    """
    accept = {"frame": mana.FRAMES, "layout": mana.LAYOUTS}
    images, seen = [], set()
    for exp_code in tqdm.tqdm(training.list_exp_codes(), desc="Read info.json", unit="expansion"):
        exp_dir = local_paths.EXP_DIR / f"exp_{exp_code}"
        image_dir = exp_dir / "image"
        on_disk = set(os.listdir(image_dir)) if image_dir.is_dir() else set()  # Faster than exists()
        if not on_disk:
            continue
        with open(exp_dir / "info.json", "r", encoding="utf-8") as fp:
            cards = json.load(fp)
        for card in cards:
            for illustration_id, _, face in download.get_card_images(card):
                if f"{illustration_id}.jpg" not in on_disk:
                    continue
                if not training._is_accepted(card, face, accept) or training._is_filtered(card, face, mana.FILTERS):
                    continue
                mana_cost = training._field_value(card, face, "mana_cost")
                if mana_cost is None or (illustration_id, mana_cost) in seen:
                    continue
                seen.add((illustration_id, mana_cost))
                images.append(CardImage(image_dir / f"{illustration_id}.jpg", exp_code, card, mana_cost))
    return images


def read_card(path: Path) -> Optional[np.ndarray]:
    """A card image resized to CARD_WIDTH x CARD_HEIGHT, None if unreadable."""
    card_bgr = cv2.imread(str(path))
    if card_bgr is None:
        print(f"  Can't read {path}")
        return None
    return cv2.resize(card_bgr, (training.CARD_WIDTH, training.CARD_HEIGHT), interpolation=cv2.INTER_LINEAR)


def top_left_fits(card_image: CardImage) -> bool:
    """True if the mana cost of the card (placed as real_window guesses it) stays out of TOP_LEFT_BOX."""
    width = row_width([HYBRID_SIZE if "/" in name else SYMBOL_SIZE for name in mana.symbols_of(card_image.mana_cost)])
    return training.CARD_WIDTH - VAL_BOX_RIGHT - width >= TOP_LEFT_BOX[2] + TOP_LEFT_MARGIN


def real_window(card_image: CardImage, classes: set) -> Tuple[Optional[np.ndarray], List[Tuple[str, Box]], str]:
    """
    VAL_RECT of a real card and the (class, box) of each symbol of its mana cost in it.

    Only the mana cost of a real card is known, not where its symbols are. The boxes are a simple
    guess, the same on every card (resources/mana_cost_symbol_boxes.json is not used on purpose):
    the symbols in a row, from right to left, the last one VAL_BOX_RIGHT pixels from the right of
    the card, every box VAL_BOX_TOP pixels from its top, sized as the generated ones (SYMBOL_SIZE,
    HYBRID_SIZE for a symbol with a '/') and SYMBOL_GAP apart. Close to the real place on most
    cards, but not exact: the classes and their order are right, so the validation must be judged on
    the mana cost read, not on the boxes (mAP, box loss).
    :return: (window, boxes, "") or (None, [], why it is skipped).
    """
    names = mana.symbols_of(card_image.mana_cost)
    if any(name not in classes for name in names):
        return None, [], "symbol not in the classes"
    x0 = int(round(VAL_RECT.center_x - VAL_RECT.width / 2))
    y0 = int(round(VAL_RECT.center_y - VAL_RECT.height / 2))
    boxes = []
    right, top = training.CARD_WIDTH - VAL_BOX_RIGHT - x0, VAL_BOX_TOP - y0  # In the window
    for name in reversed(names):
        size = HYBRID_SIZE if "/" in name else SYMBOL_SIZE
        boxes.insert(0, (name, (right - size, top, right, top + size)))
        right -= size + SYMBOL_GAP
    if any(box[0] < 0 or box[3] > IMAGE_HEIGHT for _, box in boxes):
        return None, [], "mana cost out of the window"
    card_bgr = read_card(card_image.path)
    if card_bgr is None:
        return None, [], "unreadable image"
    return training._crop_window(card_bgr, x0, y0, IMAGE_WIDTH, IMAGE_HEIGHT), boxes, ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def dataset_dir(n: int) -> Path:
    return OUTPUT_DIR / str(n)


def dataset_stamp(n: int, paths: List[Path]) -> dict:
    """What the dataset of n samples depends on (dataset.stamp), as written in json."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    stamp = {
        "version": VERSION,
        "n": n,
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "val_max": VAL_MAX,
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "mana_cost": {"parts": MANA_COST_PARTS, "numbers": number_weights(), "repeats": SYMBOL_REPEATS,
                      "distinct": DISTINCT_SYMBOLS, "order": CARD_ORDER},
        "symbol_size": SYMBOL_SIZE,
        "hybrid_size": HYBRID_SIZE,
        "symbol_gap": SYMBOL_GAP,
        "shadow": {"probability": SHADOW_PROBABILITY, "offset": list(SHADOW_OFFSET), "blur": SHADOW_BLUR,
                   "opacity": SHADOW_OPACITY},
        "excluded": EXCLUDED,
        "symbols": [path.name for path in paths],
        "symbols_hash": digest.hexdigest()[:16],
        "backgrounds": {"kinds": BACKGROUNDS, "frame": BACKGROUND_FRAME, "art_box": list(ART_BOX),
                        "top_left_box": list(TOP_LEFT_BOX), "top_left_margin": TOP_LEFT_MARGIN},
        "text": {"ratio": TEXT_RATIO, "lines": list(TEXT_LINES), "sizes": list(TEXT_SIZES),
                 "fonts": [font.name for font in text_fonts()] or "hershey"},
        "val": {"rect": list(VAL_RECT), "frames": mana.FRAMES, "layouts": mana.LAYOUTS, "filters": mana.FILTERS,
                "box_top": VAL_BOX_TOP, "box_right": VAL_BOX_RIGHT},
    }
    return json.loads(json.dumps(stamp))  # Tuples -> lists: equal to the stamp read back


class _Writer:
    """Writes the images and labels of a split (YOLO format) and their check images."""

    def __init__(self, root: Path, class_to_index: Dict[str, int]):
        self.root = root
        self.class_to_index = class_to_index
        self.entries: List[dict] = []
        self.counts: Dict[str, Counter] = {}

    def write(self, split: str, image_bgr: np.ndarray, boxes: List[Tuple[str, Box]], source: str):
        counts = self.counts.setdefault(split, Counter())
        index = counts["_images"]
        name = f"{index:06d}"
        image_dir, label_dir = self.root / "images" / split, self.root / "labels" / split
        if index == 0:
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / f"{name}.png"), image_bgr)
        with open(label_dir / f"{name}.txt", "w", encoding="utf-8") as fp:
            for box_class, (x0, y0, x1, y1) in boxes:  # No box = empty file = nothing
                fp.write(f"{self.class_to_index[box_class]} {(x0 + x1) / 2 / IMAGE_WIDTH:.6f} "
                         f"{(y0 + y1) / 2 / IMAGE_HEIGHT:.6f} {(x1 - x0) / IMAGE_WIDTH:.6f} "
                         f"{(y1 - y0) / IMAGE_HEIGHT:.6f}\n")
        mana_cost = mana.join_symbols([box_class for box_class, _ in boxes])
        self.entries.append({"file": f"images/{split}/{name}.png", "mana_cost": mana_cost,
                             "boxes": [[box_class, list(box)] for box_class, box in boxes], "source": source})
        counts["_images"] += 1
        counts["_nothing"] += not boxes
        counts.update(box_class for box_class, _ in boxes)
        if index < CHECK_IMAGES:
            self._check_image(split, name, image_bgr, boxes, mana_cost)

    def _check_image(self, split: str, name: str, image_bgr: np.ndarray, boxes: List[Tuple[str, Box]], mana_cost: str):
        """check/<split>_<name>.png: the image, CHECK_ZOOM bigger, its boxes alternating green / magenta."""
        check = cv2.resize(image_bgr, None, fx=CHECK_ZOOM, fy=CHECK_ZOOM, interpolation=cv2.INTER_NEAREST)
        for index, (_, (x0, y0, x1, y1)) in enumerate(boxes):
            color = (0, 255, 0) if index % 2 == 0 else (255, 0, 255)
            cv2.rectangle(check, (x0 * CHECK_ZOOM, y0 * CHECK_ZOOM), (x1 * CHECK_ZOOM - 1, y1 * CHECK_ZOOM - 1), color, 1)
        folder = self.root / CHECK_DIR
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{split}_{name}.png"), check)


def _print_report(counts: Dict[str, Counter], classes: List[str], skipped: Counter, root: Path):
    """Images and boxes per class of each split: printed, and saved in report.json."""
    report = {}
    for split in ("train", "val"):
        split_counts = counts.get(split, Counter())
        boxes = {name: split_counts[name] for name in classes}
        report[split] = {"images": split_counts["_images"], "nothing": split_counts["_nothing"],
                         "boxes": sum(boxes.values()), "boxes_per_class": boxes}
        present = [value for value in boxes.values() if value]
        print(f"  {split}: {split_counts['_images']} images ({split_counts['_nothing']} nothing), "
              f"{sum(boxes.values())} boxes, {len(present)}/{len(classes)} classes"
              + (f", {min(present)} to {max(present)} boxes per class" if present else ""))
    report["val_skipped"] = dict(skipped)
    if skipped:
        print(f"  val, real cards skipped: {dict(skipped)}")
    with open(root / "report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=4, ensure_ascii=False)


def build_dataset(n: int, paths: List[Path], stamp: dict) -> Tuple[Path, List[str]]:
    """Write dataset_dir(n) (removed first), dataset.stamp last. :return: (data.yaml path, classes)."""
    root = dataset_dir(n)
    if root.exists():
        shutil.rmtree(root)
    rng = random.Random(SEED)
    n_val = min(round(n * VAL_RATIO), VAL_MAX)
    n_train = n - n_val
    print(f"Mana cost dataset of {n} samples ({n_train} generated for train, {n_val} real cards for val): {root}")

    symbols = load_symbols(paths)
    check_mana_cost_symbols(symbols)
    classes = sorted(symbol.name for symbol in symbols)
    class_to_index = {name: index for index, name in enumerate(classes)}
    cards = list_card_images()
    print(f"{len(symbols)} symbols, {len(cards)} real card images.")
    writer = _Writer(root, class_to_index)

    # Val first: real cards, in a random order. They are then kept out of the training backgrounds.
    rng.shuffle(cards)
    val_paths = set()
    skipped, written = Counter(), 0
    with tqdm.tqdm(total=n_val, desc="Real cards val", unit="image") as progress:
        for card_image in cards:
            if written >= n_val:
                break
            window, boxes, why = real_window(card_image, set(classes))
            if window is None:
                skipped[why] += 1
                continue
            writer.write("val", window, boxes, str(card_image.path))
            val_paths.add(card_image.path)
            written += 1
            progress.update(1)
    if written < n_val:
        print(f"Only {written} real card images for the validation ({n_val} wanted): download more cards.")
    if not written:
        raise ValueError("No real card image for the validation.")

    # Train: generated images.
    train_cards = [card for card in cards if card.path not in val_paths]
    frame_cards = [card for card in train_cards if card.card.get("frame") == BACKGROUND_FRAME]
    pools = {"art": sorted({card.path for card in frame_cards}),
             "top_left": sorted({card.path for card in frame_cards if top_left_fits(card)})}
    for kind, pool in pools.items():
        if not pool and BACKGROUNDS.get(kind):
            raise FileNotFoundError(f"No card image of the {BACKGROUND_FRAME} frame for the '{kind}' backgrounds in "
                                    f"{local_paths.EXP_DIR}: download some first (python3 src/download.py, choice 1).")
    names = sorted({card.card["name"] for card in train_cards if card.card.get("name")})
    fonts = text_fonts()
    print(f"Backgrounds: {len(pools['art'])} cards for 'art', {len(pools['top_left'])} for 'top_left'. "
          f"Text fonts: {', '.join(font.name for font in fonts) or 'OpenCV Hershey'}.")
    kinds, weights = zip(*BACKGROUNDS.items())
    used = Counter()
    for _ in tqdm.tqdm(range(n_train), desc="Generate train", unit="image"):
        kind = rng.choices(kinds, weights)[0]
        background, source = make_background(kind, pools, rng)
        used[kind] += 1
        if rng.random() < TEXT_RATIO:
            background = draw_text(background, names, fonts, rng)
            source += " + text"
            used["text"] += 1
        image, boxes = draw_mana_cost(background, random_mana_cost(symbols, rng), rng)
        writer.write("train", image, boxes, source)
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
    _print_report(writer.counts, classes, skipped, root)
    training._write_stamp(root / STAMP_FILE, stamp)
    print(f"Mana cost dataset written: {root} (check images in {root / CHECK_DIR})")
    return root / "data.yaml", classes


def load_or_build_dataset(n: int, force: bool = False) -> Tuple[Path, List[str]]:
    """
    The dataset of n samples, built unless its dataset.stamp matches (force: always built).
    :return: (data.yaml path, classes).
    """
    if n < MIN_SAMPLES:
        raise ValueError(f"At least {MIN_SAMPLES} samples, to have a validation image: {n}")
    paths = symbol_files()
    stamp = dataset_stamp(n, paths)
    root = dataset_dir(n)
    classes_path = root / training.CLASSES_FILE
    if (not force and (root / "data.yaml").exists() and classes_path.exists()
            and training._stamp_matches(root / STAMP_FILE, stamp)):
        with open(classes_path, "r", encoding="utf-8") as fp:
            classes = json.load(fp)
        print(f"Mana cost dataset loaded from {root} ({STAMP_FILE} is up to date).")
        return root / "data.yaml", classes
    return build_dataset(n, paths, stamp)


def ask_samples() -> int:
    """Number of samples n, asked to the user."""
    while True:
        answer = input(f"How many samples (n, at least {MIN_SAMPLES}) ? : ").strip()
        if answer.isdigit() and int(answer) >= MIN_SAMPLES:
            return int(answer)
        print(f"Answer a number of at least {MIN_SAMPLES}.")


def main():
    parser = argparse.ArgumentParser(description="Dataset of the mana cost detector (see the top of the file).")
    parser.add_argument("n", nargs="?", type=int, help="number of samples (asked if not given)")
    parser.add_argument("--force", action="store_true", help="build it even if its dataset.stamp is up to date")
    args = parser.parse_args()
    load_or_build_dataset(args.n if args.n else ask_samples(), args.force)


if __name__ == "__main__":
    main()
