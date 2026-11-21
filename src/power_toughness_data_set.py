"""
Dataset of the power / toughness classifier: generated images to train, real cards to validate.

    python3 src/power_toughness_data_set.py [n] [--val PERCENT] [--nopt PERCENT] [--force]   (asked if not given)

Also built by the 'pt_gen' training of training_power_toughness.py, which trains on it.
The classifier reads the power / toughness printed at the bottom right of a straightened card: one class
per value, plus NO_VALUE ('none') for a card without power / toughness. No box (the image is the window).

n samples in <root>/runs/classify/power_toughness/gen/<n>_val<VAL>_nopt<NOPT>/, ex: gen/100000_val5_nopt10/:
VAL percent of them are real cards for the validation, the others are generated for the training, NOPT percent
of each split without power / toughness ('none').
- Classes: every power / toughness of the local scryfall data (scryfall/exp/exp_*/info.json, each face of
  each card) whose two sides match [0-9*+]+ (VALUE_PATTERN): '0/1', '2/0', '*/4', '*/1+*', '1+*/1+*', '+2/+2'
  (the augment cards of Unstable)... Not '-1/1', '?/?', '∞/5', '*/7-*'. Only values printed on real cards: no
  random value. And 'none' (NO_VALUE): no power / toughness (instant, sorcery, land, planeswalker...). The
  power / toughness of a face with its own image (transform, modal_dfc...) is its own, never the one of the
  other face (the planeswalker back of a creature has none).
- Window: IMAGE_WIDTH x IMAGE_HEIGHT (100 x 60), VAL_RECT on a CARD_WIDTH x CARD_HEIGHT card: x 364-464,
  y 592-652. Checked on the real cards: it holds the power / toughness of the frames 1993, 1997, 2003, 2015 and
  future (2015: box x 380-460 y 602-642, 2003: the same box 12 px more on the left, 1997 / 1993: white text
  ~x 408-453 y 612-642). Not there: the level up boxes (leveler), the flip cards (in the title bars), the back
  half of a meld card (a big box): EXCLUDED_LAYOUTS, never in the validation.
- train: n_train = n - n_val generated images of IMAGE_WIDTH x IMAGE_HEIGHT, the same number for every class
  (rounded, see set_symbol_data_set.even_split), in random order. Three kinds of samples (SAMPLE_KINDS, 4 : 2 : 3):
  - "plain": the background (BACKGROUNDS): one random color, or a gradient between two random colors in a random
    direction, the decorations below, the value,
  - "noise": the same background with a strong noise (NOISE_SIGMA, gray or colored), then the decorations and the
    value on top of it,
  - "old_frame": the window VAL_RECT of a real card of OLD_FRAMES (1993, 1997) without power / toughness (never a
    validation card): the textured frame where these frames print the power / toughness, then the value on it.
    No decoration (no P/T box on these frames, the crop already has the artist and copyright), the text color
    the one far enough from the frame under it (MIN_CONTRAST), the shadow forced below SHADOW_CONTRAST.
  - the value, black or white, the background colors far enough from it (MIN_CONTRAST, in luminance), in a
    random font of FONT_FILES (serif bold like the 1993 to 2003 frames, sans bold like the 2015 one), its
    digits DIGIT_HEIGHT high (13 to 23 px: 17 to 20 on the real cards), at a random place fully inside the
    image, MARGIN_X / MARGIN_Y from the edges (the zoom of the training never cuts it). On SHADOW_RATIO of them
    (one in two) a shadow of the other color, as the 1993 / 1997 frames: white text with a black shadow, black
    text with a white shadow, SHADOW_OFFSETS down right and blurred (SHADOW_BLUR). On STAR_RATIO of the values
    with a '*', '*' drawn '★' (as the 1997 and 2015 frames) in a font that has the glyph.
  - around the value, what a real card has there (without it, version 1: the model learned the text on a plain
    background only, then read the edges of the real P/T box as more characters, '4/1' -> '+4/+4', 17% of the
    real cards read right): on BOX_RATIO of them the P/T box (a rounded rectangle around the value, its fill far
    enough from the text, its outline, sometimes a second one inside: the bevel), on LINE_RATIO of them straight
    lines (frame, text box), on SMALL_TEXT_RATIO of them a small line above or below (copyright, illustrator,
    collector number). Never on the value itself. All the ratios at 0: images like version 1 (background and value).
  No position label: a classifier.
  NOPT percent of them are 'none', of the three kinds: the same images without the value (background, P/T box,
  lines, small text at the same ratios: a real card without power / toughness has there the corner of its text
  box, its frame, its copyright; "old_frame": the real frame as it is), and instead of the value (NONE_KINDS):
  nothing ("empty"), or what a card without power / toughness
  prints there: a lone number ("number", NONE_NUMBERS: the loyalty of a planeswalker, in a box at BOX_RATIO), or
  a few words of card names ("words": the end of the rules text), at the size of a value, possibly cut by the edges.
  The values share the rest, the same number each.
- val: n_val = round(n * VAL / 100) real cards (VAL_FRAMES, standard prints: see set_symbol_data_set.standard_print,
  not FILTERS, never a digital card: Arena / online only), the window VAL_RECT, their frame in images.json (the
  training prints the share read right of each frame). NOPT percent of them: cards without power / toughness,
  drawn at random (not
  NONE_EXCLUDED_LAYOUTS: oversized, art series, display fronts, minigame tokens). The others: cards with a value
  (not EXCLUDED_LAYOUTS), the same number of cards for every class, a class with fewer real cards than its share
  gives them all, the rest is split between the other classes (even_split).
  Augmented as the training images (val_augmentation, given by training_power_toughness.py: its hsv_h, hsv_s,
  hsv_v and augmentations, see set_symbol_data_set._ValAugmenter): one draw per card, seeded, made once when the
  dataset is built (the validation is never augmented by the training). None: the real cards as they are.

Files: the Ultralytics classification format: train/<folder>/*.png and val/<folder>/*.png, one folder per
class (folder_name: '2/3' -> '2_3', '*' -> 'x'), classes.json (folder -> value), images.json (each image:
"class", "source"), report.json (images per class), check/ (first images of each split, zoomed, with their
class), and dataset.stamp: the settings of the dataset, written last. The dataset is not built again while
dataset.stamp matches (--force: always). New info.json files downloaded later change the classes only with --force.
"""
import argparse
import functools
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import download
import local_paths
import set_symbol_data_set as symbol_data
import training

VERSION = 4   # Change it when the generation changes: every dataset is built again
SEED = 0      # Same settings -> same dataset

# --- Values ---
VALUE_PATTERN = re.compile(r"[0-9*+]+/[0-9*+]+")
NO_VALUE = "none"   # The class of a card without power / toughness (as search.py: mana cost 'none')
STAR = "★"   # How the 1997 frame prints '*'

# --- Window ---
IMAGE_WIDTH, IMAGE_HEIGHT = 100, 60
# Window of the real cards (validation) and of the classifier: the bottom right, x 364-464, y 592-652.
VAL_RECT = training.Rect(center_x=414, center_y=622, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
WINDOW = (int(VAL_RECT.center_x - IMAGE_WIDTH / 2), int(VAL_RECT.center_y - IMAGE_HEIGHT / 2),
          int(VAL_RECT.center_x + IMAGE_WIDTH / 2), int(VAL_RECT.center_y + IMAGE_HEIGHT / 2))

# --- Real cards (validation) ---
VAL_FRAMES = ["1993", "1997", "2003", "2015", "future"]
EXCLUDED_LAYOUTS = ["leveler", "flip", "meld"]    # Power / toughness elsewhere (see the top of the file)
# Not cards to scan without power / toughness: oversized (plane, scheme), art series, display fronts (Marvel),
# minigame and helper cards (their double_faced_token).
NONE_EXCLUDED_LAYOUTS = ["planar", "scheme", "vanguard", "art_series", "front_card", "double_faced_token"]
# Playtest cards: other frames. Type 'Card': checklists, substitutes, helper cards.
FILTERS = {"promo_types": ["playtest"], "type_line": ["Card"]}

# --- Generated images ---
SAMPLE_KINDS = {"plain": 4, "noise": 2, "old_frame": 3}   # Weights (see the top of the file)
BACKGROUNDS = {"color": 0.5, "gradient": 0.5}
NOISE_SIGMA = (15, 45)                            # "noise": standard deviation of the gaussian noise (0-255)
NOISE_COLOR_RATIO = 0.5                           # Colored noise (one per channel), else gray (the same on the 3)
OLD_FRAMES = ["1993", "1997"]                     # "old_frame": the frames of the backgrounds
TEXT_COLORS = [(0, 0, 0), (255, 255, 255)]        # Black or white (RGB)
MIN_CONTRAST = 100                                # Least luminance difference between the text and the background
DIGIT_HEIGHT = (13, 23)                           # Height of the digits (pixels)
MARGIN_X, MARGIN_Y = 10, 6                        # Text never closer to the edges (a zoom of 1.2 keeps it inside)
SHADOW_RATIO = 0.5                                # One text in two
SHADOW_OFFSETS = (1, 3)                           # Down right, pixels
SHADOW_BLUR = (0.0, 1.0)                          # Gaussian blur radius of the shadow (pixels)
SHADOW_CONTRAST = 130                             # "old_frame": below this luminance difference, always a shadow
STAR_RATIO = 0.6                                  # The 1997 and 2015 frames print '★' (checked on the real cards)
# What surrounds the value on a real card (0 = never: only the background and the value).
BOX_RATIO = 0.6                                   # The P/T box: a rounded rectangle around the value
BOX_PADDING_X, BOX_PADDING_Y = (4, 20), (2, 12)   # Space between the value and the box, each side (pixels)
BOX_OUTLINE = (1, 4)                              # Outline width (pixels)
BOX_RADIUS = (0, 14)
BOX_BEVEL_RATIO = 0.5                             # A second outline inside the first one, of another color
LINE_RATIO = 0.4                                  # 1 to LINES straight lines (frame, text box edges)
LINES = 3
LINE_WIDTH = (1, 4)
SMALL_TEXT_RATIO = 0.4                            # A small line above or below the value (copyright, illustrator)
SMALL_TEXT_SIZE = (8, 12)                         # Font size (pixels)
SMALL_TEXTS = ["™ & © {year} Wizards of the Coast", "Illus. {name}", "{number}/{total}", "© {year} Wizards",
               "{number}/{total} {rarity} {code}", "Wizards of the Coast, Inc. {number}/{total}"]
# 'none' images: what is drawn instead of the value (see the top of the file).
NONE_KINDS = {"empty": 0.5, "words": 0.35, "number": 0.15}
NONE_NUMBERS = (0, 12)                            # "number": the loyalty of a planeswalker
NONE_WORDS = (1, 3)                               # "words": words of card names
FONT_DIRS = symbol_data.TEXT_FONT_DIRS
# Lining digits only, as on the cards (not Georgia, Constantia, Candara, Corbel: short 0 1 2, 3 4 5 7 9 going down).
FONT_FILES = ["timesbd.ttf", "palab.ttf", "cambriab.ttf", "times.ttf", "pala.ttf", "DejaVuSerif-Bold.ttf",
              "framd.ttf", "trebucbd.ttf", "verdanab.ttf", "tahomabd.ttf", "calibrib.ttf", "segoeuib.ttf", "arialbd.ttf",
              "seguisym.ttf", "DejaVuSans-Bold.ttf"]

# --- Splits ---
VAL_PERCENT = 5.0       # Share of the samples for the validation (percent), proposed when it is asked
NO_PT_PERCENT = 10.0    # Share of each split without power / toughness (percent), proposed when it is asked
MIN_SAMPLES = 100

# --- Output ---
OUTPUT_DIR = local_paths.RUNS_DIR / "classify" / "power_toughness" / "gen"
CLASSES_FILE = "classes.json"
STAMP_FILE = "dataset.stamp"
CHECK_DIR = "check"
CHECK_IMAGES = 60       # First images of each split saved (zoomed, with their class) in check/
CHECK_ZOOM = 3


# ---------------------------------------------------------------------------
# Values and classes
# ---------------------------------------------------------------------------

def value_of(power, toughness) -> Optional[str]:
    """'2/3', None if there is no power / toughness or it doesn't match VALUE_PATTERN ('-1/1', '?/?'...)."""
    if power is None or toughness is None:
        return None
    value = f"{power}/{toughness}"
    return value if VALUE_PATTERN.fullmatch(value) else None


def face_power_toughness(card: dict, face: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    (power, toughness) printed on the image of a face, (None, None) without. A face with its own image
    (transform, modal_dfc...): its own only, never the one of the other face. A card with one image: the card's,
    else the one of its first face (adventure: the creature).
    """
    if face is not card:
        return face.get("power"), face.get("toughness")
    return training._raw_value(card, card, "power"), training._raw_value(card, card, "toughness")


def folder_name(value: str) -> str:
    """The class folder of a value: '*/1+*' -> 'x_1+x' ('/' can't be in a file name), 'none' -> 'none'."""
    return value.replace("/", "_").replace("*", "x")


def list_values() -> Dict[str, int]:
    """Every value of the local scryfall data (the card and each of its faces), with its number of faces."""
    counts: Counter = Counter()
    for exp_code in tqdm.tqdm(training.list_exp_codes(), desc="Read info.json", unit="expansion"):
        with open(local_paths.EXP_DIR / f"exp_{exp_code}" / "info.json", "r", encoding="utf-8") as fp:
            cards = json.load(fp)
        for card in cards:
            faces = card.get("card_faces") or [card]
            for face in faces:
                value = value_of(face.get("power"), face.get("toughness"))
                if value:
                    counts[value] += 1
    return dict(counts)


@dataclass
class RealCard:
    path: Path
    value: str      # '2/3', NO_VALUE: no power / toughness
    frame: str
    layout: str
    name: str       # The name of the face


def list_real_cards() -> List[RealCard]:
    """
    The card images on disk (VAL_FRAMES, not FILTERS, standard prints: set_symbol_data_set.standard_print, never a
    digital card: Arena / online only, "digital": true), one per
    image file: a face with a value at the place of VAL_RECT (not EXCLUDED_LAYOUTS), or without power / toughness
    (NO_VALUE, not NONE_EXCLUDED_LAYOUTS). Not: a power / toughness that doesn't match VALUE_PATTERN ('-1/1').
    """
    accept = {"frame": VAL_FRAMES}
    cards, seen = [], set()
    for exp_code in tqdm.tqdm(training.list_exp_codes(), desc="Real cards", unit="expansion"):
        exp_dir = local_paths.EXP_DIR / f"exp_{exp_code}"
        image_dir = exp_dir / "image"
        on_disk = {path.name for path in image_dir.iterdir()} if image_dir.is_dir() else set()
        if not on_disk:
            continue
        with open(exp_dir / "info.json", "r", encoding="utf-8") as fp:
            infos = json.load(fp)
        for card in infos:
            if card.get("digital"):  # Arena / online only: never printed, never scanned
                continue
            for illustration_id, _, face in download.get_card_images(card):
                path = image_dir / f"{illustration_id}.jpg"
                if path.name not in on_disk or path in seen:
                    continue
                power, toughness = face_power_toughness(card, face)
                if power is None and toughness is None:
                    value, excluded = NO_VALUE, NONE_EXCLUDED_LAYOUTS
                else:
                    value, excluded = value_of(power, toughness), EXCLUDED_LAYOUTS
                if value is None or card.get("layout") in excluded or not training._is_accepted(card, face, accept) \
                        or training._is_filtered(card, face, FILTERS) or not symbol_data.standard_print(card, face):
                    continue
                seen.add(path)
                cards.append(RealCard(path, value, card.get("frame"), card.get("layout"),
                                      face.get("name") or card.get("name") or ""))
    return cards


def real_window(card_bgr: np.ndarray) -> np.ndarray:
    """The window of a CARD_WIDTH x CARD_HEIGHT card (VAL_RECT): the image the classifier reads."""
    x0, y0, x1, y1 = WINDOW
    return card_bgr[y0:y1, x0:x1].copy()


# ---------------------------------------------------------------------------
# Generated images
# ---------------------------------------------------------------------------

def fonts() -> List[Path]:
    """The FONT_FILES found in FONT_DIRS, each one once (the first folder that has it)."""
    found = {}
    for folder in FONT_DIRS:
        for name in FONT_FILES:
            if name not in found and (Path(folder) / name).is_file():
                found[name] = Path(folder) / name
    return [found[name] for name in FONT_FILES if name in found]


@functools.lru_cache(maxsize=None)
def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _render(font: ImageFont.FreeTypeFont, text: str) -> np.ndarray:
    image = Image.new("L", (3 * font.size, 2 * font.size), 0)
    ImageDraw.Draw(image).text((font.size // 2, font.size // 4), text, fill=255, font=font)
    return np.asarray(image)


@functools.lru_cache(maxsize=None)
def has_glyph(path: Path, char: str) -> bool:
    """True if the font draws `char` (not its 'missing glyph' box: the glyph of an unassigned character)."""
    font = _font(path, 40)
    glyph = _render(font, char)
    return bool(glyph.any()) and not np.array_equal(glyph, _render(font, "\U000F0000"))


@functools.lru_cache(maxsize=None)
def _digit_height_ratio(path: Path) -> float:
    """Height of the digits / font size."""
    x0, y0, x1, y1 = _font(path, 100).getbbox("0123456789")
    return (y1 - y0) / 100


def _luminance(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _background_color(text_rgb, rng: random.Random) -> np.ndarray:
    """A random color at least MIN_CONTRAST away from the text in luminance."""
    while True:
        color = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
        if abs(_luminance(color) - _luminance(text_rgb)) >= MIN_CONTRAST:
            return color


def background(kind: str, text_rgb, rng: random.Random) -> np.ndarray:
    """IMAGE_HEIGHT x IMAGE_WIDTH RGB background: "color" (one color) or "gradient" (two colors, random direction)."""
    first = _background_color(text_rgb, rng)
    if kind == "color":
        return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), first, np.uint8)
    second = _background_color(text_rgb, rng)
    angle = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH].astype(np.float32)
    position = xs * np.cos(angle) + ys * np.sin(angle)
    position = ((position - position.min()) / (position.max() - position.min()))[:, :, None]
    return np.clip(first * (1 - position) + second * position + 0.5, 0, 255).astype(np.uint8)


def _random_rgb(rng: random.Random) -> Tuple[int, int, int]:
    return rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)


def draw_lines(draw: ImageDraw.ImageDraw, text_box: Tuple[int, int, int, int], rng: random.Random):
    """1 to LINES straight lines (horizontal or vertical, random color and width), never across the value."""
    tx0, ty0, tx1, ty1 = text_box
    for _ in range(rng.randint(1, LINES)):
        width = rng.randint(*LINE_WIDTH)
        gap = width // 2 + 2
        horizontal = rng.random() < 0.5  # Horizontal: above or below the value, vertical: left or right of it
        low, high, size = (ty0, ty1, IMAGE_HEIGHT) if horizontal else (tx0, tx1, IMAGE_WIDTH)
        spans = [span for span in [(0, low - gap), (high + gap, size - 1)] if span[0] <= span[1]]
        if not spans:
            continue
        at = rng.randint(*rng.choice(spans))
        length = IMAGE_WIDTH if horizontal else IMAGE_HEIGHT
        start, end = sorted(rng.randint(-length, 2 * length) for _ in range(2))
        points = [(start, at), (end, at)] if horizontal else [(at, start), (at, end)]
        draw.line(points, fill=_random_rgb(rng), width=width)


def draw_box(draw: ImageDraw.ImageDraw, text_box: Tuple[int, int, int, int], text_rgb, rng: random.Random):
    """
    The P/T box: a rounded rectangle around the value (may go out of the image), filled with a color far enough
    from the text (MIN_CONTRAST), its outline of a random color, on BOX_BEVEL_RATIO of them a second outline inside.
    """
    tx0, ty0, tx1, ty1 = text_box
    outline = rng.randint(*BOX_OUTLINE)
    bevel = rng.randint(1, 3) if rng.random() < BOX_BEVEL_RATIO else 0
    least = outline + bevel + 1  # The outlines never touch the value
    x0 = tx0 - max(least, rng.randint(*BOX_PADDING_X))
    x1 = tx1 + max(least, rng.randint(*BOX_PADDING_X))
    y0 = ty0 - max(least, rng.randint(*BOX_PADDING_Y))
    y1 = ty1 + max(least, rng.randint(*BOX_PADDING_Y))
    radius = min(rng.randint(*BOX_RADIUS), (x1 - x0) // 2, (y1 - y0) // 2)
    fill = tuple(int(value) for value in _background_color(text_rgb, rng))
    draw.rounded_rectangle((x0, y0, x1, y1), radius, fill=fill, outline=_random_rgb(rng), width=outline)
    if bevel:
        draw.rounded_rectangle((x0 + outline, y0 + outline, x1 - outline, y1 - outline), max(0, radius - outline),
                               outline=_random_rgb(rng), width=bevel)


_ARTISTS = ["Mark Tedin", "Rebecca Guay", "Kev Walker", "Chris Rahn", "Johannes Voss", "Ryan Pancoast", "Jesper Ejsing",
            "Terese Nielsen", "Seb McKinnon", "Anna Steinbauer"]


def draw_small_text(draw: ImageDraw.ImageDraw, text_box: Tuple[int, int, int, int], font_paths: List[Path],
                    rng: random.Random) -> bool:
    """A small line (SMALL_TEXTS) above or below the value, cut by the edges, never on it. False: no room."""
    tx0, ty0, tx1, ty1 = text_box
    text = rng.choice(SMALL_TEXTS).format(
        year=rng.randint(1993, 2026), name=rng.choice(_ARTISTS), number=rng.randint(1, 350), total=rng.randint(100, 400),
        rarity=rng.choice("CURM"), code="".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3)))
    font = _font(rng.choice(font_paths), rng.randint(*SMALL_TEXT_SIZE))
    bx0, by0, bx1, by1 = draw.textbbox((0, 0), text, font=font)
    sides = []
    if ty0 - 2 - (by1 - by0) >= -(by1 - by0) // 2:      # Above: at least half of its height visible
        sides.append(ty0 - 2 - by1)
    if ty1 + 2 <= IMAGE_HEIGHT - (by1 - by0) // 2:      # Below
        sides.append(ty1 + 2 - by0)
    if not sides:
        return False
    x = rng.randint(-(bx1 - bx0) // 2, IMAGE_WIDTH - (bx1 - bx0) // 2)
    draw.text((x, rng.choice(sides)), text, fill=rng.choice([(0, 0, 0), (255, 255, 255), _random_rgb(rng)]), font=font)
    return True


def add_noise(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    """The "noise" samples: a strong gaussian noise (NOISE_SIGMA), colored (NOISE_COLOR_RATIO) or gray."""
    noise_rng = np.random.default_rng(rng.randrange(2 ** 32))  # Seeded by rng: the same dataset every time
    shape = rgb.shape if rng.random() < NOISE_COLOR_RATIO else rgb.shape[:2] + (1,)
    noise = noise_rng.normal(0, rng.uniform(*NOISE_SIGMA), shape)
    return np.clip(rgb.astype(np.float32) + noise + 0.5, 0, 255).astype(np.uint8)


def text_color_on(image: Image.Image, text_box: Tuple[int, int, int, int], rng: random.Random
                  ) -> Tuple[Tuple[int, int, int], bool]:
    """
    "old_frame": the text color (TEXT_COLORS) far enough (MIN_CONTRAST) from the mean luminance of the frame under
    the text, at random if both are, and True if a shadow is needed (the contrast is below SHADOW_CONTRAST).
    """
    x0, y0, x1, y1 = (max(0, text_box[0]), max(0, text_box[1]), min(IMAGE_WIDTH, text_box[2]),
                      min(IMAGE_HEIGHT, text_box[3]))
    under = np.asarray(image)[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0) if x1 > x0 and y1 > y0 else (128, 128, 128)
    contrast = {color: abs(_luminance(color) - _luminance(under)) for color in TEXT_COLORS}
    good = [color for color in TEXT_COLORS if contrast[color] >= MIN_CONTRAST]
    color = rng.choice(good) if good else max(TEXT_COLORS, key=contrast.get)
    return color, contrast[color] < SHADOW_CONTRAST


def draw_shadow(image: Image.Image, position: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont,
                text_rgb, rng: random.Random) -> str:
    """The shadow of the text, of the other color (TEXT_COLORS), SHADOW_OFFSETS down right, blurred (SHADOW_BLUR)."""
    offset, radius = rng.randint(*SHADOW_OFFSETS), rng.uniform(*SHADOW_BLUR)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).text((position[0] + offset, position[1] + offset), text, fill=255, font=font)
    if radius > 0.2:
        mask = mask.filter(ImageFilter.GaussianBlur(radius))
    shadow = (255, 255, 255) if text_rgb == (0, 0, 0) else (0, 0, 0)
    image.paste(Image.new("RGB", image.size, shadow), (0, 0), mask)
    return f"shadow {offset} blur {radius:.1f}"


def generate(value: str, font_paths: List[Path], star_fonts: List[Path], words: List[str],
             old_frames: List[np.ndarray], rng: random.Random) -> Tuple[np.ndarray, str]:
    """
    A generated image of a value, or of NO_VALUE (see the top of the file). words: the words of the NO_VALUE
    "words" kind. old_frames: the BGR windows of the "old_frame" samples (empty: none of this kind).
    :return: (image BGR, description: sample kind, background, font, colors, what is drawn).
    """
    weights = {kind: weight for kind, weight in SAMPLE_KINDS.items() if kind != "old_frame" or old_frames}
    sample = rng.choices(list(weights), list(weights.values()))[0]
    text_rgb = rng.choice(TEXT_COLORS)
    if sample == "old_frame":
        kind = "old_frame"
        image = Image.fromarray(np.ascontiguousarray(rng.choice(old_frames)[:, :, ::-1]))
    else:
        kind = rng.choices(list(BACKGROUNDS), list(BACKGROUNDS.values()))[0]
        rgb = background(kind, text_rgb, rng)
        image = Image.fromarray(add_noise(rgb, rng) if sample == "noise" else rgb)
    star, none_kind = False, None
    if value == NO_VALUE:
        none_kind = rng.choices(list(NONE_KINDS), list(NONE_KINDS.values()))[0]
        if none_kind == "words":
            text = " ".join(rng.choice(words) for _ in range(rng.randint(*NONE_WORDS)))
        elif none_kind == "number":
            text = str(rng.randint(*NONE_NUMBERS))
        else:
            text = "0/0"  # "empty": only the place of a value, for the box, lines and small text. Never drawn
    else:
        star = "*" in value and bool(star_fonts) and rng.random() < STAR_RATIO
        text = value.replace("*", STAR) if star else value
    font_path = rng.choice(star_fonts if star else font_paths)
    draw = ImageDraw.Draw(image)
    size = max(6, round(rng.uniform(*DIGIT_HEIGHT) / _digit_height_ratio(font_path)))
    while True:  # Smaller until it fits (a long value: '1+*/1+*'). Words: never smaller, cut by the edges
        font = _font(font_path, size)
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        if none_kind == "words" or size <= 6 or \
                (x1 - x0 <= IMAGE_WIDTH - 2 * MARGIN_X and y1 - y0 <= IMAGE_HEIGHT - 2 * MARGIN_Y):
            break
        size -= 1
    if none_kind == "words":
        x = rng.randint(-(x1 - x0) // 2, IMAGE_WIDTH - (x1 - x0) // 2) - x0
    else:
        x = rng.randint(MARGIN_X - x0, max(MARGIN_X - x0, IMAGE_WIDTH - MARGIN_X - x1))
    y = rng.randint(MARGIN_Y - y0, max(MARGIN_Y - y0, IMAGE_HEIGHT - MARGIN_Y - y1))
    text_box = (x + x0, y + y0, x + x1, y + y1)
    need_shadow = False
    if sample == "old_frame":  # The real frame as it is: no decoration, the text color from the frame under it
        text_rgb, need_shadow = text_color_on(image, text_box, rng)
    description = f"{sample}, {kind}, {font_path.name} {size}, {'black' if text_rgb == (0, 0, 0) else 'white'}" + \
                  (f", {STAR}" if star else "") + (f", none {none_kind}" if none_kind else "")
    if sample != "old_frame":
        if rng.random() < LINE_RATIO:
            draw_lines(draw, text_box, rng)
            description += ", lines"
        if none_kind != "words" and rng.random() < BOX_RATIO:
            draw_box(draw, text_box, text_rgb, rng)
            description += ", box"
        if rng.random() < SMALL_TEXT_RATIO and draw_small_text(draw, text_box, font_paths, rng):
            description += ", small text"
    if none_kind == "empty":
        return np.ascontiguousarray(np.asarray(image)[:, :, ::-1]), description
    if need_shadow or rng.random() < SHADOW_RATIO:
        description += ", " + draw_shadow(image, (x, y), text, font, text_rgb, rng)
    ImageDraw.Draw(image).text((x, y), text, fill=text_rgb, font=font)
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1]), description




# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Sizes:
    """The number of images of a dataset of n samples."""
    train: int
    val: int
    train_none: int     # Generated images without power / toughness (NO_VALUE)
    val_none: int       # Real cards without power / toughness in the validation


def sizes(n: int, val_percent: float, no_pt_percent: float) -> Sizes:
    n_val = round(n * val_percent / 100)
    n_train = n - n_val
    return Sizes(n_train, n_val, round(n_train * no_pt_percent / 100), round(n_val * no_pt_percent / 100))


def dataset_dir(n: int, val_percent: float, no_pt_percent: float) -> Path:
    return OUTPUT_DIR / f"{n}_val{val_percent:g}_nopt{no_pt_percent:g}"


def dataset_stamp(n: int, val_percent: float, no_pt_percent: float, classes: List[str],
                  val_augmentation: Optional[dict] = None) -> dict:
    """What the dataset depends on (dataset.stamp), as written in json."""
    stamp = {
        "version": VERSION, "n": n, "seed": SEED, "val_percent": val_percent, "no_pt_percent": no_pt_percent,
        "val_augmentation": val_augmentation, "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT], "val_rect": list(VAL_RECT),
        "classes": classes, "value_pattern": VALUE_PATTERN.pattern,
        "val_cards": {"frames": VAL_FRAMES, "excluded_layouts": EXCLUDED_LAYOUTS,
                      "none_excluded_layouts": NONE_EXCLUDED_LAYOUTS, "filters": FILTERS, "digital": False,
                      "standard_print": [symbol_data.EXCLUDED_LAYOUTS, symbol_data.EXCLUDED_FRAME_EFFECTS,
                                         symbol_data.EXCLUDED_PROMO_TYPES, "borderless", "full_art", "textless",
                                         "battle"]},
        "generated": {"sample_kinds": SAMPLE_KINDS, "noise": [NOISE_SIGMA, NOISE_COLOR_RATIO], "old_frames": OLD_FRAMES,
                      "backgrounds": BACKGROUNDS, "text_colors": TEXT_COLORS, "min_contrast": MIN_CONTRAST,
                      "digit_height": DIGIT_HEIGHT, "margin": [MARGIN_X, MARGIN_Y],
                      "shadow": [SHADOW_RATIO, SHADOW_OFFSETS, SHADOW_BLUR, SHADOW_CONTRAST],
                      "star": [STAR, STAR_RATIO], "fonts": [font.name for font in fonts()],
                      "box": [BOX_RATIO, BOX_PADDING_X, BOX_PADDING_Y, BOX_OUTLINE, BOX_RADIUS, BOX_BEVEL_RATIO],
                      "lines": [LINE_RATIO, LINES, LINE_WIDTH],
                      "small_text": [SMALL_TEXT_RATIO, SMALL_TEXT_SIZE, SMALL_TEXTS],
                      "none": [NONE_KINDS, NONE_NUMBERS, NONE_WORDS]},
    }
    return json.loads(json.dumps(stamp))  # Tuples -> lists: equal to the stamp read back


class _Writer:
    """Writes the images of a split (one folder per class) and their check images."""

    def __init__(self, root: Path):
        self.root = root
        self.entries: List[dict] = []
        self.counts: Dict[str, Counter] = {}

    def write(self, split: str, image_bgr: np.ndarray, value: str, source: str, frame: Optional[str] = None):
        """frame: the frame of a real card (validation)."""
        counts = self.counts.setdefault(split, Counter())
        index = counts["_images"]
        folder = self.root / split / folder_name(value)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{index:06d}.png"
        cv2.imwrite(str(folder / name), image_bgr)
        self.entries.append({"file": f"{split}/{folder.name}/{name}", "class": value, "source": source,
                             **({"frame": frame} if frame else {})})
        counts["_images"] += 1
        counts[value] += 1
        if index < CHECK_IMAGES:
            check = cv2.resize(image_bgr, None, fx=CHECK_ZOOM, fy=CHECK_ZOOM, interpolation=cv2.INTER_NEAREST)
            cv2.putText(check, value, (3, 16), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 255), 1)
            check_dir = self.root / CHECK_DIR
            check_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(check_dir / f"{split}_{index:06d}.png"), check)


def _pick_val_cards(cards: List[RealCard], values: List[str], size: Sizes, rng: random.Random) -> List[RealCard]:
    """
    size.val real cards in random order: size.val_none cards without power / toughness drawn at random, then the
    same number of cards for every value (within the cards on disk).
    """
    by_class: Dict[str, List[RealCard]] = defaultdict(list)
    for card in sorted(cards, key=lambda card: str(card.path)):
        by_class[card.value].append(card)
    for value in sorted(by_class):
        rng.shuffle(by_class[value])
    nones = by_class[NO_VALUE][:size.val_none]
    if len(nones) < size.val_none:
        raise ValueError(f"{size.val_none} validation cards without power / toughness wanted, only {len(nones)} on "
                         f"disk: download more cards or lower the percent.")
    counts = symbol_data.even_split(size.val - size.val_none, {value: len(by_class[value]) for value in values}, rng)
    picked = nones + [card for value in values for card in by_class[value][:counts[value]]]
    rng.shuffle(picked)
    missing = [value for value in values if not by_class[value]]
    if missing:
        print(f"Val: {len(missing)} classes without real card (only generated, never validated): {', '.join(missing)}")
    return picked


def _print_report(counts: Dict[str, Counter], classes: List[str], extra: dict, root: Path):
    report = {}
    for split in ("train", "val"):
        split_counts = counts.get(split, Counter())
        per_class = {value: split_counts[value] for value in classes}
        present = [count for value, count in per_class.items() if count and value != NO_VALUE]
        report[split] = {"images": split_counts["_images"], "none": split_counts[NO_VALUE], "per_class": per_class}
        print(f"  {split}: {split_counts['_images']} images, {split_counts[NO_VALUE]} without power / toughness, "
              f"{len(present)}/{len(classes) - 1} values" + (f", {min(present)} to {max(present)} per value" if present else ""))
    report.update(extra)
    with open(root / "report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=4, ensure_ascii=False)


def build_dataset(n: int, val_percent: float, no_pt_percent: float, classes: List[str], cards: List[RealCard],
                  stamp: dict, val_augmentation: Optional[dict] = None) -> Tuple[Path, Dict[str, str]]:
    """
    Write dataset_dir(...) (removed first), dataset.stamp last. classes: the values, then NO_VALUE.
    val_augmentation: see set_symbol_data_set.val_augmentation_of (None: the real cards as they are).
    :return: (dataset folder, folder -> value).
    """
    root = dataset_dir(n, val_percent, no_pt_percent)
    if root.exists():
        shutil.rmtree(root)
    rng = random.Random(SEED)
    size = sizes(n, val_percent, no_pt_percent)
    values = [value for value in classes if value != NO_VALUE]
    font_paths = fonts()
    if not font_paths:
        raise FileNotFoundError(f"No font of FONT_FILES in {FONT_DIRS}: install one (ex: sudo apt install fonts-dejavu).")
    star_fonts = [path for path in font_paths if has_glyph(path, STAR)]
    # The "words" of the images without power / toughness: the words of the card names.
    words = sorted({word for card in cards for word in re.findall(r"[A-Za-z']{2,}", card.name)})
    print(f"Power / toughness dataset of {n} samples: {size.train} generated for train ({size.train_none} without "
          f"power / toughness), {size.val} real cards for val ({size.val_none} without): {root}")
    print(f"{len(classes)} classes, {len(cards)} real card images ({sum(card.value == NO_VALUE for card in cards)} "
          f"without power / toughness). Fonts: {', '.join(path.name for path in font_paths)} "
          f"('{STAR}': {', '.join(path.name for path in star_fonts) or 'none'}).")
    writer = _Writer(root)

    # Val: real cards.
    val_cards = _pick_val_cards(cards, values, size, rng)
    augment = symbol_data._ValAugmenter(val_augmentation) if val_augmentation else None
    frames = Counter()
    for card in tqdm.tqdm(val_cards, desc="Real cards val", unit="image"):
        card_bgr = symbol_data.read_card(card.path)
        if card_bgr is None:
            continue
        window = real_window(card_bgr)
        if augment is not None:
            window = augment(window)
        writer.write("val", window, card.value, str(card.path), card.frame)
        frames[card.frame] += 1
    if len(val_cards) < size.val:
        print(f"Only {len(val_cards)} real card images for the validation ({size.val} wanted): download more cards.")
    if not val_cards:
        raise ValueError("No real card image for the validation.")
    print(f"Val frames: {dict(frames)}")

    # The backgrounds of the "old_frame" samples: the windows of the old cards without power / toughness, never
    # a validation card.
    val_paths = {card.path for card in val_cards}
    old_cards = [card for card in cards if card.value == NO_VALUE and card.frame in OLD_FRAMES
                 and card.path not in val_paths]
    with ThreadPoolExecutor(8) as pool:
        old_frames = [real_window(card_bgr) for card_bgr in tqdm.tqdm(
            pool.map(lambda card: symbol_data.read_card(card.path), old_cards), total=len(old_cards),
            desc="Old frames", unit="image") if card_bgr is not None]
    if not old_frames:
        print(f"No card of {OLD_FRAMES} without power / toughness on disk: no \"old_frame\" sample.")
    print(f"Old frame backgrounds: {len(old_frames)} windows of {', '.join(OLD_FRAMES)} cards without power / toughness.")

    # Train: generated images, size.train_none without power / toughness, the rest the same number for every value.
    counts = symbol_data.even_split(size.train - size.train_none, {value: None for value in values}, rng)
    labels = [NO_VALUE] * size.train_none + [value for value in values for _ in range(counts[value])]
    rng.shuffle(labels)
    used = Counter()
    for value in tqdm.tqdm(labels, desc="Generate train", unit="image"):
        image, description = generate(value, font_paths, star_fonts, words, old_frames, rng)
        writer.write("train", image, value, description)
        used[description.split(",")[0]] += 1
        for extra in ("shadow", STAR, "box", "lines", "small text") + tuple(f"none {kind}" for kind in NONE_KINDS):
            used[extra] += extra in description
    print(f"Train images: {dict(used)}")

    folders = {folder_name(value): value for value in classes}
    with open(root / CLASSES_FILE, "w", encoding="utf-8") as fp:
        json.dump(dict(sorted(folders.items())), fp, indent=4, ensure_ascii=False)
    with open(root / "images.json", "w", encoding="utf-8") as fp:
        json.dump(writer.entries, fp, indent=4, ensure_ascii=False)
    _print_report(writer.counts, classes, {"val_frames": dict(frames), "train_used": dict(used)}, root)
    training._write_stamp(root / STAMP_FILE, stamp)
    print(f"Power / toughness dataset written: {root} (check images in {root / CHECK_DIR})")
    return root, folders


def check_sizes(n: int, val_percent: float, no_pt_percent: float = 0.0) -> Optional[str]:
    """Why these settings can't make a dataset, None if they can."""
    if n < MIN_SAMPLES:
        return f"at least {MIN_SAMPLES} samples"
    if not 0 < val_percent < 100:
        return "a validation percent between 0 and 100"
    if not 0 <= no_pt_percent < 100:
        return "a percent without power / toughness from 0 to less than 100"
    size = sizes(n, val_percent, no_pt_percent)
    if size.val < 1:
        return f"at least one validation image: {val_percent:g}% of {n} is less than 1"
    if size.train < 1:
        return "at least one generated image for the training"
    if size.val - size.val_none < 1 or size.train - size.train_none < 1:
        return "at least one image with a power / toughness in each split"
    return None


def load_or_build_dataset(n: int, val_percent: float = VAL_PERCENT, no_pt_percent: float = NO_PT_PERCENT,
                          force: bool = False, val_augmentation: Optional[dict] = None) -> Tuple[Path, Dict[str, str]]:
    """
    The dataset of n samples, val_percent% of them real cards for the validation, no_pt_percent% of each split
    without power / toughness, built unless its dataset.stamp matches (force: always). val_augmentation: the
    augmentation of the real cards of the validation (set_symbol_data_set.val_augmentation_of), None: none.
    :return: (dataset folder, folder -> value).
    """
    problem = check_sizes(n, val_percent, no_pt_percent)
    if problem:
        raise ValueError(f"{n} samples, {val_percent:g}% of validation, {no_pt_percent:g}% without power / "
                         f"toughness: {problem}")
    root = dataset_dir(n, val_percent, no_pt_percent)
    classes_path = root / CLASSES_FILE
    if not force and classes_path.exists() and (root / STAMP_FILE).exists():
        with open(classes_path, "r", encoding="utf-8") as fp:
            folders = json.load(fp)
        classes = sorted(value for value in folders.values() if value != NO_VALUE) + [NO_VALUE]
        if training._stamp_matches(root / STAMP_FILE, dataset_stamp(n, val_percent, no_pt_percent, classes,
                                                                    val_augmentation)):
            print(f"Power / toughness dataset loaded from {root} ({STAMP_FILE} is up to date).")
            return root, folders
    values = list_values()
    if not values:
        raise ValueError(f"No power / toughness in {local_paths.EXP_DIR}: download cards first (python3 src/download.py).")
    classes = sorted(values) + [NO_VALUE]
    print(f"{len(values)} values, from {sum(values.values())} faces: {', '.join(sorted(values))}, and '{NO_VALUE}'")
    cards = list_real_cards()
    return build_dataset(n, val_percent, no_pt_percent, classes, cards,
                         dataset_stamp(n, val_percent, no_pt_percent, classes, val_augmentation), val_augmentation)


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
        percent = symbol_data._ask_percent("Percent of the samples for the validation (real cards)", VAL_PERCENT)
        if percent is None:
            continue
        problem = check_sizes(n, percent)
        if problem is None:
            size = sizes(n, percent, 0)
            print(f"Validation: {size.val} real cards, training: {size.train} generated images.")
            return percent
        print(f"Not possible: {problem}.")


def ask_no_pt_percent(n: int, val_percent: float) -> float:
    """Percent of each split without power / toughness, asked to the user (Enter: NO_PT_PERCENT)."""
    while True:
        percent = symbol_data._ask_percent("Percent of the images without power / toughness (train and val)",
                                           NO_PT_PERCENT)
        if percent is None:
            continue
        problem = check_sizes(n, val_percent, percent)
        if problem is None:
            size = sizes(n, val_percent, percent)
            print(f"Without power / toughness: {size.val_none} real cards of the validation, {size.train_none} "
                  f"generated images of the training.")
            return percent
        print(f"Not possible: {problem}.")


def main():
    parser = argparse.ArgumentParser(description="Dataset of the power / toughness classifier (see the top of the file).")
    parser.add_argument("n", nargs="?", type=int, help="number of samples (asked if not given)")
    parser.add_argument("--val", type=float, help="percent of the samples for the validation (asked if not given)")
    parser.add_argument("--nopt", type=float, help="percent of each split without power / toughness (asked if not given)")
    parser.add_argument("--force", action="store_true", help="build it even if its dataset.stamp is up to date")
    args = parser.parse_args()
    n = args.n if args.n else ask_samples()
    val_percent = args.val if args.val is not None else ask_val_percent(n)
    no_pt_percent = args.nopt if args.nopt is not None else ask_no_pt_percent(n, val_percent)
    import training_power_toughness  # The augmentation of the pt_gen training, also given to the validation
    load_or_build_dataset(n, val_percent, no_pt_percent, args.force,
                          training_power_toughness.val_augmentation(training_power_toughness.definitions()[0]))


if __name__ == "__main__":
    main()
