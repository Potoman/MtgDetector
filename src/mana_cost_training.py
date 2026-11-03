"""
Mana cost detection: a YOLO detector that finds each mana symbol of the mana cost, at the top right
of a straightened card. Uses the steps of training.py (keys, downloads, stamps, YOLO dataset and
training), with its own:

- Boxes: one box and one class per {...} symbol of the mana cost: '{1}{W}{W}' -> 3 boxes, classes
  '1', 'W', 'W', right aligned, in the order of info.json (see SymbolBox). The place and the size of
  the symbols change with the frame and the expansion: resources/mana_cost_symbol_boxes.json gives
  one SymbolBox per expansion, measured by symbol_offsets.py (run it first, and again when new
  expansions are downloaded).
- Card selection: the same number of boxes for every symbol (boxes_per_class), whatever its
  frequency ({G} is on 14000 card images, {13} on 6):
  1. pool: every card image (samples_per_key, see training.collect_keys) of an accepted card,
  2. selection, rarest symbol first: different card images are added until the symbol has enough
     boxes, preferring the cards whose other symbols already have enough,
  3. split train / val (a card image of a rare symbol always stays in train),
  4. the train images of the symbols still under boxes_per_class are duplicated (train.txt lists
     them several times, the images are written once). The validation is not duplicated.
  The result is printed and saved in <hash>/dataset/balance.json.
- Answer of the model: the symbols found, joined from left to right: '{1}{W}{W}' (read_mana_cost).

Files: runs/detect/mana_cost/<name>/<hash>/, as every TrainingYoloDetect (see training.py).
Run from anywhere: python3 src/mana_cost_training.py (or from the menu of training.py)
"""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import tqdm

import download
import training
from training import Box, DetectImage, ImageRef, Rect, TrainingYoloDetect

# ---------------------------------------------------------------------------
# Cards (shared with symbol_offsets.py)
# ---------------------------------------------------------------------------

FRAMES = ["2003", "2015"]
LAYOUTS = ["normal", "flip", "transform", "modal_dfc", "leveler"]
ACCEPT = {"frame": FRAMES, "layout": LAYOUTS}
# Planeswalkers: other frame. Playtest cards (unk, cmb1, cmb2, mb2...): placeholder art or mana cost elsewhere.
FILTERS = {"type_line": ["Legendary Planeswalker", "Planeswalker"], "promo_types": ["playtest"], "set": ["otp"]}

# ---------------------------------------------------------------------------
# Symbols and their boxes (shared with symbol_offsets.py)
# ---------------------------------------------------------------------------


class SymbolBox(NamedTuple):
    """
    Row of right-aligned mana symbols, in pixel coordinates of a CARD_WIDTH x CARD_HEIGHT card.
    Each {...} of the mana cost is one symbol, with its own box and its own class (the text between
    the braces): '{1}{W}{W}' -> '1', 'W', 'W'. The symbols are placed from left to right in the order
    of the mana cost, the last one touching the right edge of the row. A symbol is symbol_width wide
    (+ the hybrid extra width for a hybrid one) and `height` high.
    """
    symbol_width: float     # Width of one symbol (ex: 253 / 10)
    height: float           # Height of the boxes
    right: float            # Distance between the right edge of the last box and the right border of the card
    top: float              # Distance between the top of the boxes and the top of the card


# Set by hand on the 2015 frame. The reference of symbol_offsets.py.
DEFAULT_SYMBOL_BOX = SymbolBox(symbol_width=253 / 10, height=28, right=35, top=37)
SYMBOL_BOXES_FILE = "resources/mana_cost_symbol_boxes.json"  # From <root>, written by symbol_offsets.py

SYMBOL_PATTERN = re.compile(r"\{[^}]+\}")
# Hybrid mana, ex: {W/U}, {2/R}, {G/W/P}: a symbol with a '/' that is not a one-color Phyrexian mana
# ({U/P}, {C/P}), which is drawn at the normal size.
HYBRID_SYMBOL_PATTERN = re.compile(r"\{(?![A-Za-z0-9]+/P\})[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+\}")
# Extra width (pixels) of a hybrid symbol of DEFAULT_SYMBOL_BOX.symbol_width, scaled with the symbol width.
HYBRID_EXTRA_WIDTH = 5


def symbols_of(value: str) -> List[str]:
    """Classes of a mana cost, from left to right: '{2}{W/U}' -> ['2', 'W/U']."""
    return [symbol[1:-1] for symbol in SYMBOL_PATTERN.findall(value)]


def join_symbols(classes: List[str]) -> str:
    """Inverse of symbols_of: ['1', 'W', 'W'] -> '{1}{W}{W}'."""
    return "".join(f"{{{name}}}" for name in classes)


def symbol_boxes_in_card(symbol_box: SymbolBox, value: str) -> List[Tuple[str, Box]]:
    """
    One (class, box) per {...} symbol of the mana cost `value`, from left to right (see SymbolBox).
    The edges are computed in float then rounded, so two neighbor boxes share the same edge.
    """
    symbols = SYMBOL_PATTERN.findall(value)
    if not symbols:
        raise ValueError(f"No {{...}} symbol in {value!r}")
    y0, y1 = int(round(symbol_box.top)), int(round(symbol_box.top + symbol_box.height))
    hybrid_extra = HYBRID_EXTRA_WIDTH * symbol_box.symbol_width / DEFAULT_SYMBOL_BOX.symbol_width
    right = training.CARD_WIDTH - symbol_box.right
    boxes = []
    for symbol in reversed(symbols):  # From the right edge, which doesn't move
        width = symbol_box.symbol_width + (hybrid_extra if HYBRID_SYMBOL_PATTERN.fullmatch(symbol) else 0)
        left = right - width
        boxes.append((symbol[1:-1], (int(round(left)), y0, int(round(right)), y1)))
        right = left
    return boxes[::-1]


def lookup_keys(exp_code: str, frame: str, set_type: str) -> List[str]:
    """
    Keys of the symbol boxes json for a card, in the order they are tried: the expansion, the
    expansion + frame (an expansion mixing frames), every card of that frame and set type, every card
    of that frame (see symbol_offsets.py).
    """
    return [exp_code, f"{exp_code}:{frame}", f"_type_{frame}_{set_type}", f"_frame_{frame}"]


_symbol_boxes_cache: Dict[str, dict] = {}


def load_symbol_boxes(path: str) -> dict:
    """The "boxes" of the json of symbol_offsets.py, <root>/<path> (cached)."""
    if path not in _symbol_boxes_cache:
        with open(training.ROOT / path, "r", encoding="utf-8") as fp:
            _symbol_boxes_cache[path] = json.load(fp)["boxes"]
    return _symbol_boxes_cache[path]


def symbol_box_of_image(definition: "ManaCostTraining", image: ImageRef) -> Tuple[SymbolBox, str]:
    """SymbolBox of a card image and the json key used ("symbol_box" = the definition default)."""
    if definition.symbol_boxes is None:
        return definition.symbol_box, "symbol_box"
    boxes = load_symbol_boxes(definition.symbol_boxes)
    card = download._get_card(image.exp_code, image.card_id)
    for key in lookup_keys(image.exp_code, card.get("frame"), card.get("set_type", "")):
        if key in boxes:
            values = boxes[key]
            return SymbolBox(values["symbol_width"], values["height"], values["right"], values["top"]), key
    return definition.symbol_box, "symbol_box"


# ---------------------------------------------------------------------------
# Training definition
# ---------------------------------------------------------------------------

@dataclass
class ManaCostTraining(TrainingYoloDetect):
    """
    TrainingYoloDetect settings (see training.py), plus the ones below. label must be 'mana_cost'.
    samples_per_key is the size of the pool: card images per mana cost value, the selection is made
    in it. ALL_IMAGES = every card image (one per illustration and mana cost).
    """
    symbol_box: SymbolBox = DEFAULT_SYMBOL_BOX      # Cards the json doesn't know
    symbol_boxes: Optional[str] = SYMBOL_BOXES_FILE  # None = symbol_box for every card
    # Wanted number of boxes of each symbol in the training split, with the duplicated images.
    boxes_per_class: int = 300
    # Maximum number of times a card image is used by the training. 0 = no limit (a symbol on a
    # single card image: that image is used boxes_per_class times).
    max_copies_per_image: int = 0

    KIND = "YOLO detect, mana cost"

    def extra_fields(self) -> dict:
        return {
            "type": "mana_cost",
            "symbol_box": list(self.symbol_box),
            "symbol_boxes": _file_hash(self.symbol_boxes) if self.symbol_boxes is not None else None,
            "boxes_per_class": self.boxes_per_class,
            "max_copies_per_image": self.max_copies_per_image,
            "balance_version": BALANCE_VERSION,
        }

    def run(self) -> Optional[Path]:
        return process_mana_cost_training(self)


BALANCE_VERSION = 1  # Change it when the selection below changes, to build the dataset again


def _file_hash(path: str) -> str:
    """A new symbol boxes json gives a new <hash> and a new dataset."""
    return hashlib.sha256((training.ROOT / path).read_bytes()).hexdigest()[:16]


ALL_IMAGES = 1_000_000  # samples_per_key: no limit


def definitions() -> List[ManaCostTraining]:
    """The mana cost trainings, also listed in the menu of training.py."""
    return [
        # Read at the top right of the card. Lands (no mana cost) are "nothing" images. The rect is
        # the search window (jitter 0: same window for every card); translate moves it at random
        # at each epoch, as a card a bit off after reshape_card.
        ManaCostTraining(
            name="mana_cost",
            label="mana_cost",
            sub_categories=[],
            rect=Rect(center_x=334, center_y=51, height=102, width=308),
            accept=ACCEPT,
            filters=FILTERS,
            samples_per_key=ALL_IMAGES,
            boxes_per_class=300,
            input_size=320,
            jitter=0.0,
            epochs=15,
            patience=5,  # The validation stops improving after ~10-13 epochs
            batch_size=32,
            blur_kernels=[7, 9],
            cache="ram",
            translate=0.15,
            mosaic=1.0,
            # Webcam colors: white balance, dull or no colors (a {G} seen grey). In gray, the model
            # must read the shape (tree, skull, drop...). Only on part of the training images: the
            # Phyrexian symbols ({G/P}, {U/P}...) differ only by their color. The validation images
            # and the webcam keep their colors.
            augmentations=[
                ("PlanckianJitter", {"p": 0.5}),  # White balance (light color temperature)
                ("RandomBrightnessContrast", {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5}),
                ("RandomGamma", {"gamma_limit": (80, 120), "p": 0.3}),
                ("ToGray", {"p": 0.4}),  # Last, so that these images stay fully gray
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Balanced selection
# ---------------------------------------------------------------------------

MIN_IMAGES_FOR_VAL = 5  # A symbol on fewer selected card images keeps all of them in train


@dataclass
class Pick:
    """One card image of the selection."""
    key: Tuple[str, ...]
    image: ImageRef
    symbols: List[str]      # Its classes, from left to right (with repeats: '{W}{W}' -> ['W', 'W'])


def _class_order(counts: Counter) -> List[str]:
    """Rarest class first."""
    return sorted(counts, key=lambda name: (counts[name], name))


def select_balanced(keys: Dict[Tuple[str, ...], List[ImageRef]], target: int,
                    rng: random.Random) -> List[Pick]:
    """
    Different card images, until every symbol has `target` boxes (or no other card image has it).
    Rarest symbol first; among the card images of a symbol, those whose other symbols already have
    `target` boxes come last (less overshoot of the frequent symbols).
    """
    pool = [Pick(key, image, symbols_of(key[0])) for key, images in sorted(keys.items()) if key[0]
            for image in images]
    rng.shuffle(pool)
    by_class: Dict[str, List[int]] = defaultdict(list)
    pool_counts = Counter()
    for index, pick in enumerate(pool):
        pool_counts.update(pick.symbols)
        for name in set(pick.symbols):
            by_class[name].append(index)

    counts, chosen = Counter(), set()
    for name in _class_order(pool_counts):
        candidates = [i for i in by_class[name] if i not in chosen]
        candidates.sort(key=lambda i: sum(1 for s in pool[i].symbols if s != name and counts[s] >= target))
        for index in candidates:
            if counts[name] >= target:
                break
            chosen.add(index)
            counts.update(pool[index].symbols)
    return [pool[i] for i in sorted(chosen)]


def split_picks(picks: List[Pick], val_ratio: float, rng: random.Random) -> Tuple[List[Pick], List[Pick]]:
    """Random split; a card image with a symbol on fewer than MIN_IMAGES_FOR_VAL card images stays in train."""
    images_per_class = Counter(name for pick in picks for name in set(pick.symbols))
    train, val = [], []
    for pick in picks:
        can_val = all(images_per_class[name] >= MIN_IMAGES_FOR_VAL for name in pick.symbols)
        (val if can_val and rng.random() < val_ratio else train).append(pick)
    return train, val


def balance_copies(train: List[Pick], target: int, max_copies: int) -> List[int]:
    """
    How many times each train card image is used, so that every symbol reaches `target` boxes.
    Rarest symbol first; the least used card image of the symbol is added again (ties: the one whose
    other symbols already have `target` boxes last).
    """
    copies = [1] * len(train)
    counts = Counter(name for pick in train for name in pick.symbols)
    by_class: Dict[str, List[int]] = defaultdict(list)
    for index, pick in enumerate(train):
        for name in set(pick.symbols):
            by_class[name].append(index)
    for name in _class_order(counts):
        while counts[name] < target:
            usable = [i for i in by_class[name] if not max_copies or copies[i] < max_copies]
            if not usable:
                break
            index = min(usable, key=lambda i: (copies[i], sum(1 for s in train[i].symbols
                                                              if s != name and counts[s] >= target)))
            copies[index] += 1
            counts.update(train[index].symbols)
    return copies


def _balance_report(definition: ManaCostTraining, pool_counts: Counter, train: List[Pick], copies: List[int],
                    val: List[Pick], dataset_dir: Path):
    """Boxes per class: pool, train (different images), train (with the copies), val. Printed and saved."""
    distinct = Counter(name for pick in train for name in pick.symbols)
    used = Counter()
    for pick, n in zip(train, copies):
        for name in pick.symbols:
            used[name] += n
    in_val = Counter(name for pick in val for name in pick.symbols)
    rows = [{"class": name, "pool": pool_counts[name], "train_images": distinct[name], "train": used[name],
             "val": in_val[name]} for name in _class_order(pool_counts)]
    print(f"[{definition.name}] Boxes per class (target {definition.boxes_per_class} in train):")
    print(f"  {'class':<6} {'pool':>7} {'train (different)':>18} {'train (copies)':>15} {'val':>5}")
    for row in rows:
        print(f"  {row['class']:<6} {row['pool']:>7} {row['train_images']:>18} {row['train']:>15} {row['val']:>5}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / "balance.json", "w", encoding="utf-8") as fp:
        json.dump({"boxes_per_class": definition.boxes_per_class, "train_images": len(train),
                   "train_images_with_copies": sum(copies), "val_images": len(val), "classes": rows}, fp, indent=4)


# ---------------------------------------------------------------------------
# Dataset and training
# ---------------------------------------------------------------------------

def _symbol_images(definition: ManaCostTraining, picks: List[Pick], rng: random.Random,
                   cut_values: Counter, box_keys: Counter) -> Tuple[List[Optional[DetectImage]], Dict[int, np.ndarray]]:
    """One window per pick (None if its image is missing), and the card images by pick index (for the backgrounds)."""
    images, cards = [], {}
    for index, pick in enumerate(tqdm.tqdm(picks, desc=f"[{definition.name}] Read cards", unit="card")):
        card_bgr = training.read_card(pick.image)
        if card_bgr is None:
            images.append(None)
            continue
        cards[index] = card_bgr
        x, y = training.random_window(definition, rng)
        symbol_box, box_key = symbol_box_of_image(definition, pick.image)
        box_keys[box_key if box_key.startswith("_") or box_key == "symbol_box" else "<expansion>"] += 1
        boxes, cut = training.boxes_in_window(definition, symbol_boxes_in_card(symbol_box, pick.key[0]),
                                              x, y, pick.key[0])
        cut_values[pick.key[0]] += cut
        images.append(DetectImage(training.cut_window(definition, card_bgr, x, y), pick.key[0], boxes,
                                  pick.key, str(training._image_path(pick.image))))
    return images, cards


def _background_images(definition: ManaCostTraining, count: int, lands: List[Tuple[Tuple[str, ...], ImageRef]],
                       picks: List[Pick], cards: Dict[int, np.ndarray], rng: random.Random) -> List[DetectImage]:
    """
    `count` "nothing" images: half the search window of lands (no mana cost, the most realistic
    case), the rest anywhere else on the selected cards (see training.background_image).
    """
    images = []
    for key, image in tqdm.tqdm(lands[:count // 2], desc=f"[{definition.name}] Background cards", unit="card"):
        card_bgr = training.read_card(image)
        if card_bgr is not None:
            x, y = training.random_window(definition, rng)
            images.append(DetectImage(training.cut_window(definition, card_bgr, x, y), None, [], key,
                                      str(training._image_path(image))))
    indexes = sorted(cards)
    for _ in range(count - len(images)):
        if not indexes:
            break
        index = rng.choice(indexes)
        background = training.background_image(definition, cards[index], picks[index].key,
                                                str(training._image_path(picks[index].image)), rng)
        if background is not None:
            images.append(background)
    return images


def load_or_build_dataset(definition: ManaCostTraining, keys: Dict[Tuple[str, ...], List[ImageRef]]
                          ) -> Optional[Tuple[Path, List[str]]]:
    """Balanced <hash>/dataset/, built unless dataset.stamp matches. :return: (data.yaml path, classes)."""
    stamp = training._dataset_stamp(definition, keys)
    loaded = training.load_detect_dataset(definition, stamp)
    if loaded is not None:
        return loaded
    dataset_dir = training.clear_detect_dataset(definition)
    rng = random.Random(training.DATASET_SEED)

    picks = select_balanced(keys, round(definition.boxes_per_class / (1 - definition.val_ratio)), rng)
    lands = [(key, image) for key, images in sorted(keys.items()) if not key[0] for image in images]
    rng.shuffle(lands)
    n_backgrounds = round(definition.background_ratio * len(picks))
    # Only the selected card images are downloaded (the pool is much bigger).
    selected: Dict[Tuple[str, ...], List[ImageRef]] = defaultdict(list)
    for pick in picks:
        selected[pick.key].append(pick.image)
    for key, image in lands[:n_backgrounds // 2]:
        selected[key].append(image)
    training.download_images(definition, dict(selected))

    cut_values, box_keys = Counter(), Counter()
    images, cards = _symbol_images(definition, picks, rng, cut_values, box_keys)
    missing = sum(image is None for image in images)
    kept = [(pick, image) for pick, image in zip(picks, images) if image is not None]
    print(f"[{definition.name}] {len(kept)} card images selected from {sum(len(v) for v in keys.values())} "
          f"(pool), SymbolBox used: {dict(box_keys.most_common())}")
    training.print_cut_warning(definition, cut_values)

    train_picks, val_picks = split_picks([pick for pick, _ in kept], definition.val_ratio, rng)
    image_of = {id(pick): image for pick, image in kept}
    copies = balance_copies(train_picks, definition.boxes_per_class, definition.max_copies_per_image)
    pool_counts = Counter(name for key, images_ in keys.items() if key[0] for _ in images_ for name in symbols_of(key[0]))
    _balance_report(definition, pool_counts, train_picks, copies, val_picks, dataset_dir)

    backgrounds = _background_images(definition, n_backgrounds, lands, picks, cards, rng)
    n_val_backgrounds = int(len(backgrounds) * definition.val_ratio)
    train = [image_of[id(pick)] for pick in train_picks] + backgrounds[n_val_backgrounds:]
    val = [image_of[id(pick)] for pick in val_picks] + backgrounds[:n_val_backgrounds]
    copies += [1] * (len(train) - len(copies))
    return training.write_detect_dataset(definition, stamp, train, val, rng, missing, copies)


def process_mana_cost_training(definition: ManaCostTraining) -> Optional[Path]:
    training.start_training(definition)
    keys = training.load_or_collect_keys(definition)
    print(f"[{definition.name}] {len(keys)} mana cost values in the pool.")
    dataset = load_or_build_dataset(definition, keys)
    if dataset is None:
        print(f"[{definition.name}] No symbol image, skipped.")
        return None
    yaml_path, classes = dataset
    return training.train_yolo_detect(definition, yaml_path, classes, extra_info={"detector": "mana_cost"})


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def is_mana_cost_detector(bundle: dict) -> bool:
    """True for a model of this file (detector.json of training.load_detector)."""
    return bundle.get("detector") == "mana_cost" or bundle.get("symbol_box") is not None  # symbol_box: first version


def read_mana_cost(bundle: dict, card_bgr: np.ndarray, min_confidence: float = 0.25
                   ) -> Tuple[Optional[str], float, List[Box]]:
    """
    The mana cost of a straightened card: the symbols found, joined from left to right ('{1}{W}{W}').
    :return: (mana cost, lowest confidence of its symbols, box of each symbol in card coordinates),
             (None, 0.0, []) if no symbol was found.
    """
    # agnostic_nms: a symbol has only one class.
    found = training.detect_boxes(bundle, card_bgr, min_confidence, agnostic_nms=True)
    if not found:
        return None, 0.0, []
    return (join_symbols([name for name, _, _ in found]), min(conf for _, conf, _ in found),
            [box for _, _, box in found])


if __name__ == "__main__":
    for chosen in training._choose_trainings(definitions()):
        chosen.run()
