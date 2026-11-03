"""Generic image-classification trainings on a rectangle of the card.

A training is defined by:
- a label: the card field the model has to learn (ex: 'rarity'),
- a set of sub-categories: card fields that influence the image (ex: 'set'),
- a rectangle (center_x, center_y, height, width) in pixel coordinates of a card resized
  to CARD_WIDTH x CARD_HEIGHT. (0, 0) is the top-left of the card, 'x' is horizontal and
  'y' increases toward the bottom of the card.

For each training:
1. Every <root>/scryfall/exp_<exp_code>/info.json is read. For every card image (each face
   of a double-faced card is its own image), the label value and the sub-category values
   build a key: (label, sub_category_1, sub_category_2, ...).
2. Each unique key is kept once (up to `samples_per_key` different images).
3. The images of each key are downloaded if needed (download.download_if_not_exist).
4. The rectangle is cropped. The crop is bound to the label value.
5. A small PyTorch CNN classifier is trained on the crops and saved in
   <root>/runs/classify/<label>/<name>/<hash>/model.pt

Files of a TrainingDefinition:
  <root>/runs/classify/<label>/crops/          crops saved as PNG to check the rectangle (save_crops),
                                               and the crops used by the training:
                                               crops_<...>.npz + crops_<...>.stamp, shared by every
                                               TrainingDefinition that crops the same way
  <root>/runs/classify/<label>/<name>/<hash>/  one folder per training: training.json (the settings
                                               of this training), keys.stamp, download.stamp,
                                               keys.json, model.pt, samples.json
  <hash> is computed on the TrainingDefinition fields that change the trained model (see
  _training_fields): changing epochs, blur, shifts... gives a new folder, so every training
  is kept. The same settings always give the same <hash>.

TrainingYoloDetect: same settings (except shift_pixels), but step 4-5 build a YOLO detection
dataset and train an Ultralytics YOLO detector instead of the basic CNN. The detector answers
"no symbol" (nothing found) or the label value of the symbol it found (ex: the expansion code),
wherever the symbol is in a search window around the rectangle. Files:
  <root>/runs/detect/<label>/<name>/<hash>/    training.json, keys.stamp, download.stamp, keys.json,
                                               dataset/ (YOLO dataset + dataset.stamp), check/
                                               (save_crops), yolo/ (Ultralytics run), model.pt,
                                               detector.json
The keys and downloads are shared with the TrainingDefinitions of the same label.

Steps 1-2, 3 and 4 are stamped: keys.stamp, download.stamp and crops_<...>.stamp. A step whose stamp
matches the TrainingDefinition is skipped on the next launch. Delete a .stamp file to force its
step (and the following ones) to run again.

Uses PyTorch only. Do not import TensorFlow (or modules that import it, such as
algo_corner_detection, dataset or algo) from this file: loading TensorFlow and PyTorch in
the same process can make their CUDA libraries clash.

Run from anywhere: python3 src/training.py
"""
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

import download


ROOT = download.ROOT
SCRYFALL_DIR = download.SCRYFALL_DIR
OUTPUT_DIR = ROOT / "runs" / "classify"
DETECT_OUTPUT_DIR = ROOT / "runs" / "detect"  # TrainingYoloDetect
CROPS_PNG_DIR = "crops"  # Name of the PNG crops folder, in runs/classify/<label>/

# Reference card size of the rectangle coordinates (size of a scryfall 'normal' image).
CARD_WIDTH = 488
CARD_HEIGHT = 680


class Rect(NamedTuple):
    """Rectangle in pixel coordinates of a CARD_WIDTH x CARD_HEIGHT card."""
    center_x: float
    center_y: float
    height: float
    width: float


class SymbolBox(NamedTuple):
    """
    Box of a row of right-aligned symbols (ex: the mana cost), in pixel coordinates of a
    CARD_WIDTH x CARD_HEIGHT card. The number of symbols n is the number of {...} in the label
    value ('{0}{R}' -> 2): the box is n x symbol_width wide, and its right edge doesn't move.
    """
    symbol_width: float     # Width of one symbol (ex: 253 / 10)
    height: float           # Height of the box
    right: float            # Distance between the right edge of the box and the right border of the card
    top: float              # Distance between the top of the box and the top of the card


SYMBOL_PATTERN = re.compile(r"\{[^}]+\}")
HYBRID_SYMBOL_PATTERN = re.compile(r"\{[A-Za-z0-9]+/[A-Za-z0-9]+\}")   # Hybrid mana, ex: {W/U}, {2/R}
HYBRID_EXTRA_WIDTH = 6      # Extra width (pixels) of the box for each hybrid mana symbol


def symbol_box_in_card(symbol_box: SymbolBox, value: str) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the symbols of `value` in the card (see SymbolBox)."""
    count = len(SYMBOL_PATTERN.findall(value))
    if count == 0:
        raise ValueError(f"No {{...}} symbol in {value!r}: symbol_box needs one per symbol")
    hybrid_count = len(HYBRID_SYMBOL_PATTERN.findall(value))
    x1 = CARD_WIDTH - symbol_box.right
    y0 = symbol_box.top
    width = count * symbol_box.symbol_width + hybrid_count * HYBRID_EXTRA_WIDTH
    return (int(round(x1 - width)), int(round(y0)),
            int(round(x1)), int(round(y0 + symbol_box.height)))


@dataclass
class TrainingDefinition:
    name: str                       # Output folder name: runs/classify/<name>
    label: str                      # Card field to learn (ex: 'rarity')
    sub_categories: List[str]       # Card fields that influence the image (ex: ['set'])
    rect: Rect                      # Area of the card to classify
    # --- Optional ---
    exp_codes: Optional[List[str]] = None   # None = every scryfall/exp_*/info.json
    # Card field -> accepted values. Empty = every card is accepted. Otherwise a card image is
    # accepted only if, for every field, its value is in the list (for a list field such as
    # 'frame_effects', at least one of its values must be in the list).
    accept: Dict[str, List[str]] = field(default_factory=dict)
    # Card field -> rejected values. Empty = nothing is rejected. Otherwise a card image is
    # rejected if, for any field, its value is in the list (for a list field, if any of its
    # values is in the list). Applied after `accept`.
    filters: Dict[str, List[str]] = field(default_factory=dict)
    samples_per_key: int = 1        # Number of different images kept for each unique key
    input_size: int = 64            # The crop is resized to input_size x input_size
    jitter: float = 0.1             # Random shift of the rectangle (fraction of its size) for augmentation
    epochs: int = 400
    batch_size: int = 32
    learning_rate: float = 1e-3
    val_ratio: float = 0.2          # Only applied to labels having at least 2 samples
    save_crops: bool = True         # Save the crops in runs/classify/<name>/crops to check the rectangle
    # Gaussian blur kernel sizes (odd numbers, an even size is rounded up). Empty = no blur, the
    # samples are used as they are. Otherwise every sample is replaced by one copy per size, and
    # 1 is the original sample (not blurred): [1, 5] doubles the dataset, [1, 3, 7] triples it,
    # [3] only keeps the blurred copies (the originals are dropped).
    blur_kernels: List[int] = field(default_factory=list)
    # Shifted copies: for each distance of the list, every sample is moved by that many pixels
    # (card coordinates) in the 8 directions (straight and diagonal). The uncovered side repeats
    # the border pixels. Each distance gives 8 more samples per sample:
    # [4] -> dataset x9, [3, 6, 9] -> dataset x25. Empty = no shifted copies.
    shift_pixels: List[int] = field(default_factory=list)


@dataclass
class TrainingYoloDetect:
    """
    Same settings as TrainingDefinition (see its comments), without shift_pixels: the model is a
    YOLO detector (Ultralytics) instead of the basic CNN. It finds the symbol wherever it is in a
    search window, so shifted copies are not needed.

    Search window = the rectangle + `jitter` x its size on each side (rect 64x64, jitter 0.5 ->
    window 128x128). In each training image the window is moved at random, so the symbol can be
    anywhere in it (always fully visible). Windows cut where there is no symbol are added as
    "nothing" images (background_ratio).
    Answer of the model: nothing found = no symbol, otherwise the label value of the best box.
    """
    name: str                       # Output folder name: runs/detect/<label>/<name>
    label: str                      # Card field to learn (ex: 'set'). One YOLO class per value
    sub_categories: List[str]       # Card fields that influence the image (ex: ['rarity'])
    rect: Rect                      # Area of the card where the symbol is (the box of the labels)
    # --- Optional ---
    # None = the box of every image is the rect. Otherwise one box per image, computed from its
    # label value (see SymbolBox), ex: the mana cost '{2}{G}' -> a box 2 symbols wide. The rect
    # stays the search area (same window for every card): it must contain the boxes, a box going
    # out of the window is cut at its border (counted and printed).
    symbol_box: Optional[SymbolBox] = None
    exp_codes: Optional[List[str]] = None
    accept: Dict[str, List[str]] = field(default_factory=dict)
    filters: Dict[str, List[str]] = field(default_factory=dict)
    samples_per_key: int = 1
    input_size: int = 128           # YOLO imgsz (multiple of 32): the search window is resized to it
    jitter: float = 0.5             # Search window margin on each side, as a fraction of the rect size
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3     # Initial learning rate of the AdamW optimizer (YOLO lr0)
    val_ratio: float = 0.2          # Only applied to labels having at least 2 samples
    save_crops: bool = True         # Save the windows with their box in <hash>/check/ to check the rect
    # Gaussian blur kernel sizes, as in TrainingDefinition: 1 = the original image. [1, 3] keeps
    # every image and adds its blurred copy, [3] only keeps the blurred copies. The downscale
    # copies are always made from the original image, with or without 1.
    blur_kernels: List[int] = field(default_factory=list)
    # Low resolution copies, like a small card seen by the webcam then straightened by reshape_card:
    # each factor (0-1) adds one copy of every image shrunk by that factor, with a little noise and
    # JPEG compression, then enlarged back to its size. Ex: [0.35, 0.45, 0.6]. Empty = no copies.
    downscale: List[float] = field(default_factory=list)
    # --- YOLO only ---
    model: str = "yolo26n.pt"       # Pretrained detector: <root>/<model> if it exists, else downloaded
    background_ratio: float = 0.25  # Number of "nothing" images, as a fraction of the symbol images
    workers: int = 4                # Processes preparing the images in parallel (0 = all in the main process)
    patience: int = 10              # Stop when the validation score didn't improve for this many epochs
    # Image cache (Ultralytics): False = read, decode and resize every image again at each epoch,
    # "ram" = done once, the images are kept in memory (no disk access after), "disk" = done once,
    # saved as .npy files next to the images.
    cache: Union[bool, str] = False
    # Ultralytics augmentations that move the symbol, redone at every epoch. 0.0 = disabled.
    translate: float = 0.1          # Random shift, as a fraction of the image size (+/-)
    scale: float = 0.2              # Random zoom (+/-)
    mosaic: float = 1.0             # Probability to combine 4 images into one


# ---------------------------------------------------------------------------
# 1. Search the keys in scryfall/exp_*/info.json
# ---------------------------------------------------------------------------

@dataclass
class ImageRef:
    exp_code: str
    card_id: str
    illustration_id: str


def list_exp_codes() -> List[str]:
    return sorted(p.parent.name[len("exp_"):] for p in SCRYFALL_DIR.glob("exp_*/info.json"))


def _normalize(value) -> Optional[str]:
    """Turn a card field value into a class name. None means 'not available'."""
    if value is None:
        return None
    if isinstance(value, list):
        # ex: colors ['U', 'B'] -> 'BU', [] -> 'none'
        return "".join(sorted(str(v) for v in value)) or "none"
    return str(value)


def _raw_value(card: dict, face: dict, name: str):
    """Raw value of a field for one image: the face value first, then the card value."""
    if face is not card and name in face:
        return face[name]
    if name in card:
        return card[name]
    # Single image card with faces (split, adventure, flip...): some fields only exist in faces.
    faces = card.get("card_faces") or []
    if faces and name in faces[0]:
        return faces[0][name]
    return None


def _field_value(card: dict, face: dict, name: str) -> Optional[str]:
    return _normalize(_raw_value(card, face, name))


def _filter_value(card: dict, face: dict, name: str):
    """
    Value of a field used by `accept` and `filters`.
    'type_line' keeps only the part before the em dash:
    'Legendary Planeswalker — Karn' -> 'Legendary Planeswalker'.
    """
    value = _raw_value(card, face, name)
    if name == "type_line" and isinstance(value, str):
        value = value.split("—")[0].strip()
    return value


def _as_list(values) -> List[str]:
    """Allow a single string instead of a list: "Instant" -> ["Instant"]."""
    return [values] if isinstance(values, str) else values


def _is_accepted(card: dict, face: dict, accept: Dict[str, List[str]]) -> bool:
    """Empty `accept` = everything is accepted. Otherwise every field must have an accepted value."""
    for name, allowed in accept.items():
        allowed = _as_list(allowed)
        value = _filter_value(card, face, name)
        if value is None:
            return False
        if isinstance(value, list):
            if not any(str(v) in allowed for v in value):
                return False
        elif str(value) not in allowed:
            return False
    return True


def _is_filtered(card: dict, face: dict, filters: Dict[str, List[str]]) -> bool:
    """Empty `filters` = nothing is rejected. Otherwise rejected if any field has a filtered value."""
    for name, rejected in filters.items():
        rejected = _as_list(rejected)
        value = _filter_value(card, face, name)
        if value is None:
            continue
        if isinstance(value, list):
            if any(str(v) in rejected for v in value):
                return True
        elif str(value) in rejected:
            return True
    return False


def collect_keys(definition: TrainingDefinition) -> Dict[Tuple[str, ...], List[ImageRef]]:
    """
    :return: for each unique key (label, *sub_categories), up to `samples_per_key` images.
    """
    fields = [definition.label] + list(definition.sub_categories)
    keys: Dict[Tuple[str, ...], List[ImageRef]] = defaultdict(list)

    for exp_code in definition.exp_codes or list_exp_codes():
        info_path = SCRYFALL_DIR / f"exp_{exp_code}" / "info.json"
        if not info_path.exists():
            print(f"[{definition.name}] Missing {info_path}, skipped.")
            continue
        with open(info_path, "r", encoding="utf-8") as fp:
            cards = json.load(fp)

        for card in cards:
            for illustration_id, image_url, face in download.get_card_images(card):
                if not image_url:
                    continue
                if not _is_accepted(card, face, definition.accept):
                    continue
                if _is_filtered(card, face, definition.filters):
                    continue
                values = tuple(_field_value(card, face, name) for name in fields)
                if None in values:
                    continue  # The card doesn't have the label or a sub-category
                images = keys[values]
                if (len(images) < definition.samples_per_key
                        and all(image.illustration_id != illustration_id for image in images)):
                    images.append(ImageRef(exp_code, card["id"], illustration_id))
    return dict(keys)


# Stamps, in runs/classify/<label>/<name>/. Each one holds what its step depends on: if it matches the
# current TrainingDefinition, the step is skipped. Each stamp is removed before its step runs
# and written last, so a step stopped in the middle is never seen as done.
# - keys.stamp: the TrainingDefinition fields used to list the keys (keys saved in keys.json).
# - download.stamp: the same fields + a hash of the keys. Only written if every image was
#   downloaded, so failed downloads are retried at the next launch.
# - crops_<...>.stamp, in runs/classify/<label>/crops/: the same fields + a hash of the keys +
#   rect and jitter, which change the crops (crops saved in crops_<...>.npz next to it).
#   The file name is built from these values (see _crops_base_name), so every TrainingDefinition
#   that crops the same way uses the same files, whatever its name, shifts, blur, epochs...
# The hash of the keys makes download.stamp and the crops stamp invalid when the keys are
# collected again and are different, even with the same fields (ex: new info.json files).
KEYS_FILE = "keys.json"
KEYS_STAMP_FILE = "keys.stamp"
OLD_KEYS_STAMP_FILE = "done.stamp"  # Previous name of keys.stamp, renamed automatically
DOWNLOAD_STAMP_FILE = "download.stamp"


def _output_root(definition) -> Path:
    """runs/classify/ for a TrainingDefinition, runs/detect/ for a TrainingYoloDetect."""
    return DETECT_OUTPUT_DIR if isinstance(definition, TrainingYoloDetect) else OUTPUT_DIR


def _label_dir(definition: TrainingDefinition) -> Path:
    """runs/classify/<label>/ (runs/detect/<label>/ for a TrainingYoloDetect)"""
    return _output_root(definition) / _safe(definition.label)


def _name_dir(definition: TrainingDefinition) -> Path:
    """runs/classify/<label>/<name>/: one <hash>/ folder per training."""
    return _label_dir(definition) / definition.name


TRAINING_FILE = "training.json"  # Settings of a training, in its <hash>/ folder


def _training_fields(definition: TrainingDefinition) -> dict:
    """
    The TrainingDefinition fields that change the trained model, used for <hash>:
    - the keys: label, sub_categories, and what selects the cards (exp_codes, accept, filters),
    - samples_per_key, epochs, batch_size, blur_kernels, shift_pixels,
    - rect, jitter (the crops), input_size, learning_rate, val_ratio.
    Not used: name (already in the path), save_crops (only writes the PNG check images),
    workers and cache (only change the speed).
    A TrainingYoloDetect has no shift_pixels, and adds its type, model and background_ratio.
    """
    fields = {
        "label": definition.label,
        "sub_categories": list(definition.sub_categories),
        "exp_codes": definition.exp_codes,
        "accept": {name: _as_list(values) for name, values in definition.accept.items()},
        "filters": {name: _as_list(values) for name, values in definition.filters.items()},
        "samples_per_key": definition.samples_per_key,
        "epochs": definition.epochs,
        "batch_size": definition.batch_size,
        **_blur_stamp(definition.blur_kernels),
        "rect": list(definition.rect),
        "jitter": definition.jitter,
        "input_size": definition.input_size,
        "learning_rate": definition.learning_rate,
        "val_ratio": definition.val_ratio,
    }
    if isinstance(definition, TrainingYoloDetect):
        fields.update({"type": "yolo_detect", "model": definition.model,
                       "background_ratio": definition.background_ratio, "patience": definition.patience})
        if definition.downscale:  # Only when used: the trainings without it keep their <hash>
            fields["downscale"] = list(definition.downscale)
        if definition.symbol_box is not None:  # Only when used, as downscale
            fields["symbol_box"] = list(definition.symbol_box)
        # Only when changed: the trainings with the default augmentations keep their <hash>
        for name, default in (("translate", 0.1), ("scale", 0.2), ("mosaic", 1.0)):
            if getattr(definition, name) != default:
                fields[name] = getattr(definition, name)
    else:
        fields["shift_pixels"] = (definition.shift_pixels if isinstance(definition.shift_pixels, int)
                                  else list(definition.shift_pixels))
    return fields


def _training_hash(definition: TrainingDefinition) -> str:
    """First 10 characters of the hash of _training_fields."""
    content = json.dumps(_training_fields(definition), sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]


def _output_dir(definition: TrainingDefinition) -> Path:
    """runs/classify/<label>/<name>/<hash>/: training.json, stamps, keys.json, model.pt, samples.json."""
    return _name_dir(definition) / _training_hash(definition)


def _crops_png_dir(definition: TrainingDefinition) -> Path:
    """runs/classify/<label>/crops/: the PNG crops (save_crops)."""
    return _label_dir(definition) / CROPS_PNG_DIR


# Folders of previous versions: runs/classify/<name>/ and runs/classify/<label>/<name>/.
def _old_dirs(definition: TrainingDefinition) -> List[Path]:
    new_dir = _output_dir(definition)
    return [d for d in (OUTPUT_DIR / definition.name, _name_dir(definition)) if d.is_dir() and d != new_dir]


def _reuse_from_other_training(definition: TrainingDefinition, stamp_name: str, stamp: dict,
                               data_files: List[str]) -> bool:
    """
    Copy a valid stamp (and its data files) from another training of the same label, so a new
    <hash>/ folder (ex: only epochs changed) doesn't collect the keys or check the downloads again.
    Trainings of runs/classify/ and runs/detect/ share their keys and download stamps.
    """
    output_dir = _output_dir(definition)
    other_stamps = sorted(stamp for root in (OUTPUT_DIR, DETECT_OUTPUT_DIR)
                          for stamp in (root / _safe(definition.label)).glob(f"*/*/{stamp_name}"))
    for other_stamp in other_stamps:
        other_dir = other_stamp.parent
        if other_dir == output_dir or not all((other_dir / f).is_file() for f in data_files):
            continue
        if _stamp_matches(other_stamp, stamp):
            output_dir.mkdir(parents=True, exist_ok=True)
            for file_name in data_files:
                shutil.copy2(other_dir / file_name, output_dir / file_name)
            shutil.copy2(other_stamp, output_dir / stamp_name)  # Stamp last
            print(f"[{definition.name}] {stamp_name} reused from {other_dir}")
            return True
    return False


def _stamp_matches(stamp_path: Path, stamp: dict) -> bool:
    """True if the stamp file exists and holds exactly `stamp`."""
    try:
        with open(stamp_path, "r", encoding="utf-8") as fp:
            return json.load(fp) == stamp
    except (OSError, ValueError):
        return False


def _remove_stamp(stamp_path: Path):
    if stamp_path.exists():
        stamp_path.unlink()


def _write_stamp(stamp_path: Path, stamp: dict):
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stamp_path, "w", encoding="utf-8") as fp:
        json.dump(stamp, fp, indent=4)


def _keys_hash(keys: Dict[Tuple[str, ...], List["ImageRef"]]) -> str:
    content = json.dumps([[list(key), [vars(image) for image in images]] for key, images in sorted(keys.items())])
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _keys_stamp(definition: TrainingDefinition) -> dict:
    """The TrainingDefinition fields that change the result of collect_keys."""
    return {
        "label": definition.label,
        "sub_categories": list(definition.sub_categories),
        "exp_codes": definition.exp_codes,
        "accept": {name: _as_list(values) for name, values in definition.accept.items()},
        "filters": {name: _as_list(values) for name, values in definition.filters.items()},
        "samples_per_key": definition.samples_per_key,
    }


def load_or_collect_keys(definition: TrainingDefinition) -> Dict[Tuple[str, ...], List[ImageRef]]:
    """collect_keys, or the saved keys if keys.stamp matches the current TrainingDefinition."""
    output_dir = _output_dir(definition)
    keys_path = output_dir / KEYS_FILE
    stamp_path = output_dir / KEYS_STAMP_FILE
    stamp = _keys_stamp(definition)

    # Stamp written by a previous version under its old name: rename it once.
    old_stamp_path = output_dir / OLD_KEYS_STAMP_FILE
    if old_stamp_path.exists() and not stamp_path.exists():
        old_stamp_path.rename(stamp_path)

    if not (keys_path.exists() and _stamp_matches(stamp_path, stamp)):
        _reuse_from_other_training(definition, KEYS_STAMP_FILE, stamp, [KEYS_FILE])

    if keys_path.exists() and _stamp_matches(stamp_path, stamp):
        try:
            with open(keys_path, "r", encoding="utf-8") as fp:
                saved_keys = json.load(fp)
            keys = {tuple(entry["key"]): [ImageRef(**image) for image in entry["images"]]
                    for entry in saved_keys}
            print(f"[{definition.name}] Keys loaded from {keys_path} ({KEYS_STAMP_FILE} is up to date).")
            return keys
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"[{definition.name}] Can't read the saved keys ({error}), collecting them again.")
    elif stamp_path.exists():
        print(f"[{definition.name}] {KEYS_STAMP_FILE} doesn't match the TrainingDefinition, collecting the keys again.")

    _remove_stamp(stamp_path)
    keys = collect_keys(definition)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(keys_path, "w", encoding="utf-8") as fp:
        json.dump([{"key": list(key), "images": [vars(image) for image in images]}
                   for key, images in keys.items()], fp, indent=4)
    _write_stamp(stamp_path, stamp)
    return keys


# ---------------------------------------------------------------------------
# 2. Download the images
# ---------------------------------------------------------------------------

def _image_path(image: ImageRef) -> Path:
    """Where download.download_if_not_exist saves the image of this face."""
    return SCRYFALL_DIR / f"exp_{image.exp_code}" / "image" / f"{image.illustration_id}.jpg"


def _download_stamp(definition: TrainingDefinition, keys: Dict[Tuple[str, ...], List[ImageRef]]) -> dict:
    return {"keys": _keys_stamp(definition), "keys_hash": _keys_hash(keys)}


def download_images(definition: TrainingDefinition, keys: Dict[Tuple[str, ...], List[ImageRef]]):
    """Download every image of the keys that is not on disk yet, unless download.stamp matches."""
    stamp_path = _output_dir(definition) / DOWNLOAD_STAMP_FILE
    stamp = _download_stamp(definition, keys)
    if not _stamp_matches(stamp_path, stamp):
        _reuse_from_other_training(definition, DOWNLOAD_STAMP_FILE, stamp, [])
    if _stamp_matches(stamp_path, stamp):
        print(f"[{definition.name}] Images already downloaded ({DOWNLOAD_STAMP_FILE} is up to date).")
        return

    _remove_stamp(stamp_path)
    images = [image for key in sorted(keys) for image in keys[key]]
    failed = []
    for image in tqdm.tqdm(images, desc=f"[{definition.name}] Download", unit="image"):
        # Download all faces of the card if needed, then check the face of this key.
        paths = download.download_if_not_exist(image.exp_code, image.card_id)
        if str(_image_path(image)) not in paths:
            failed.append(_image_path(image))

    if failed:
        print(f"[{definition.name}] {len(failed)} image(s) could not be downloaded, ex: {failed[0]}. "
              f"{DOWNLOAD_STAMP_FILE} not written: they will be retried at the next launch.")
        return
    _write_stamp(stamp_path, stamp)


# ---------------------------------------------------------------------------
# 3. Crop the rectangle
# ---------------------------------------------------------------------------

def _crop_margin(rect: Rect, jitter: float) -> Tuple[int, int]:
    return int(round(rect.width * jitter)), int(round(rect.height * jitter))


def crop_rect(card_bgr: np.ndarray, rect: Rect, margin_x: int = 0, margin_y: int = 0) -> np.ndarray:
    """
    Crop the rectangle (+ margin on each side) of a card. The card is first resized to
    CARD_WIDTH x CARD_HEIGHT. Pixels outside the card replicate the border.
    """
    card_bgr = cv2.resize(card_bgr, (CARD_WIDTH, CARD_HEIGHT), interpolation=cv2.INTER_LINEAR)
    x0 = int(round(rect.center_x - rect.width / 2)) - margin_x
    y0 = int(round(rect.center_y - rect.height / 2)) - margin_y
    x1 = x0 + int(round(rect.width)) + 2 * margin_x
    y1 = y0 + int(round(rect.height)) + 2 * margin_y

    pad = max(0, -x0, -y0, x1 - CARD_WIDTH, y1 - CARD_HEIGHT)
    if pad:
        card_bgr = cv2.copyMakeBorder(card_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return card_bgr[y0 + pad:y1 + pad, x0 + pad:x1 + pad].copy()


@dataclass
class Sample:
    crop_bgr: np.ndarray    # Rectangle + jitter margin
    label: str
    key: Tuple[str, ...]
    image_path: str


def build_samples(definition: TrainingDefinition,
                  keys: Dict[Tuple[str, ...], List[ImageRef]]) -> List[Sample]:
    margin_x, margin_y = _crop_margin(definition.rect, definition.jitter)
    crops_dir = _crops_png_dir(definition)

    samples: List[Sample] = []
    for index, (key, images) in enumerate(sorted(keys.items())):
        print(f"[{definition.name}] Key {index + 1}/{len(keys)}: {key}")
        for image in images:
            # The images are downloaded by download_images, before this step.
            expected_path = _image_path(image)
            if not expected_path.exists():
                print(f"  Image not available: {expected_path}")
                continue

            card_bgr = cv2.imread(str(expected_path))
            if card_bgr is None:
                print(f"  Unreadable image: {expected_path}")
                continue

            crop = crop_rect(card_bgr, definition.rect, margin_x, margin_y)
            samples.append(Sample(crop, key[0], key, str(expected_path)))

            if definition.save_crops:
                label_dir = crops_dir / "exp" / f"exp_{_safe(key[0])}"
                label_dir.mkdir(parents=True, exist_ok=True)
                name = "_".join(_safe(v) for v in key[1:]) or "key"
                cv2.imwrite(str(label_dir / f"{name}_{image.illustration_id[:8]}.png"),
                            crop[margin_y:crop.shape[0] - margin_y, margin_x:crop.shape[1] - margin_x])
    return samples


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def _crop_stamp(definition: TrainingDefinition, keys: Dict[Tuple[str, ...], List[ImageRef]]) -> dict:
    """What the crops depend on: the keys, and rect + jitter (the jitter sets the margin kept around the rectangle)."""
    return {
        "keys": _keys_stamp(definition),
        "keys_hash": _keys_hash(keys),
        "rect": list(definition.rect),
        "jitter": definition.jitter,
    }


def _number(value) -> str:
    """436 -> '436', 430.5 -> '430.5', 0.1 -> '0.1'"""
    return f"{value:g}"


def _crops_base_name(definition: TrainingDefinition, stamp: dict) -> str:
    """
    Name of the crops files, built from everything used to make them:
      crops_<label>_<sub_categories>_n<samples_per_key>_r<x>-<y>-<height>x<width>_j<jitter>_<hash>
    ex: crops_set_rarity-colors_n10_r436-402-64x64_j0.1_3fa9c2d81e
    exp_codes, accept, filters and the keys themselves are too long for a file name: they are
    in <hash>, the first 10 characters of the hash of the whole stamp (also written in the
    .stamp file in full). Two different stamps always give two different names.
    """
    rect = definition.rect
    digest = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return "_".join([
        "crops",
        _safe(definition.label),
        "-".join(_safe(name) for name in definition.sub_categories) or "none",
        f"n{definition.samples_per_key}",
        f"r{_number(rect.center_x)}-{_number(rect.center_y)}-{_number(rect.height)}x{_number(rect.width)}",
        f"j{_number(definition.jitter)}",
        digest,
    ])


def load_or_build_samples(definition: TrainingDefinition,
                          keys: Dict[Tuple[str, ...], List[ImageRef]]) -> List[Sample]:
    """
    build_samples, or the saved crops if runs/classify/<label>/crops/crops_<...>.stamp matches the
    current TrainingDefinition and keys. The crops files are shared by every TrainingDefinition
    that crops the same way.
    """
    stamp = _crop_stamp(definition, keys)
    crops_dir = _crops_png_dir(definition)
    base_name = _crops_base_name(definition, stamp)
    crops_path = crops_dir / f"{base_name}.npz"
    stamp_path = crops_dir / f"{base_name}.stamp"

    if crops_path.exists() and _stamp_matches(stamp_path, stamp):
        try:
            with np.load(crops_path, allow_pickle=False) as data:
                crops, labels = data["crops"], data["labels"]
                sample_keys, image_paths = data["keys"], data["image_paths"]
                samples = [Sample(crops[i], str(labels[i]), tuple(json.loads(str(sample_keys[i]))), str(image_paths[i]))
                           for i in range(len(labels))]
            print(f"[{definition.name}] {len(samples)} crops loaded from {crops_path} ({stamp_path.name} is up to date).")
            return samples
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"[{definition.name}] Can't read the saved crops ({error}), cropping the images again.")
    else:
        print(f"[{definition.name}] No saved crops for these settings ({base_name}), cropping the images.")

    _remove_stamp(stamp_path)
    samples = build_samples(definition, keys)
    missing = sum(len(images) for images in keys.values()) - len(samples)
    if not samples:
        return samples
    crops_dir.mkdir(parents=True, exist_ok=True)
    # All the crops of a TrainingDefinition have the same size: one array (N, H, W, 3).
    with open(crops_path, "wb") as fp:
        np.savez_compressed(
            fp,
            crops=np.stack([sample.crop_bgr for sample in samples]),
            labels=np.array([sample.label for sample in samples]),
            keys=np.array([json.dumps(list(sample.key)) for sample in samples]),
            image_paths=np.array([sample.image_path for sample in samples]),
        )
    if missing:
        # Some images were missing or unreadable: no stamp, the cropping is done again next time.
        print(f"[{definition.name}] {missing} image(s) missing, {stamp_path.name} not written.")
        return samples
    _write_stamp(stamp_path, stamp)
    return samples


# ---------------------------------------------------------------------------
# 3. PyTorch dataset and model
# ---------------------------------------------------------------------------

def _to_tensor(crop_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    image = cv2.resize(crop_bgr, (input_size, input_size), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)


class CropDataset(Dataset):
    def __init__(self, samples: List[Sample], class_to_index: Dict[str, int],
                 definition: TrainingDefinition, augment: bool):
        self.samples = samples
        self.class_to_index = class_to_index
        self.input_size = definition.input_size
        self.margin_x, self.margin_y = _crop_margin(definition.rect, definition.jitter)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        crop = sample.crop_bgr
        height = crop.shape[0] - 2 * self.margin_y
        width = crop.shape[1] - 2 * self.margin_x

        if self.augment:
            # Random shift of the rectangle inside the margin
            dx = random.randint(0, 2 * self.margin_x)
            dy = random.randint(0, 2 * self.margin_y)
            crop = crop[dy:dy + height, dx:dx + width]
            # Random brightness / contrast
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-25, 25)
            crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        else:
            crop = crop[self.margin_y:self.margin_y + height, self.margin_x:self.margin_x + width]

        return _to_tensor(crop, self.input_size), self.class_to_index[sample.label]


class SimpleClassifier(nn.Module):
    """Basic CNN: 3 conv blocks + global average pooling + linear classifier."""

    def __init__(self, num_classes: int):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split(samples: List[Sample], val_ratio: float) -> Tuple[List[Sample], List[Sample]]:
    """Split per label so that every label keeps at least one training sample."""
    by_label: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)
    train, val = [], []
    for label_samples in by_label.values():
        random.shuffle(label_samples)
        n_val = int(len(label_samples) * val_ratio)
        val.extend(label_samples[:n_val])
        train.extend(label_samples[n_val:])
    return train, val


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            correct += (model(images).argmax(dim=1) == targets).sum().item()
            total += targets.numel()
    return correct / total if total else float("nan")


# (dx, dy) of the 8 directions: diagonals and straight.
SHIFT_DIRECTIONS = [(-1, -1), (0, -1), (1, -1),
                    (-1, 0),           (1, 0),
                    (-1, 1),  (0, 1),  (1, 1)]


def _add_shifted_copies(samples: List[Sample], shifts: List[int]) -> List[Sample]:
    """
    Return the samples + 8 copies of every sample for each distance of `shifts`, moved by that
    many pixels in the 8 directions (border pixels repeated).
    """
    if isinstance(shifts, int):  # Also accept a single distance: 4 -> [4]
        shifts = [shifts]
    copies = []
    for sample in samples:
        height, width = sample.crop_bgr.shape[:2]
        for shift in shifts:
            if shift <= 0:
                continue
            for dx, dy in SHIFT_DIRECTIONS:
                matrix = np.float32([[1, 0, dx * shift], [0, 1, dy * shift]])
                shifted = cv2.warpAffine(sample.crop_bgr, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
                copies.append(Sample(shifted, sample.label, sample.key, sample.image_path))
    return samples + copies


def _blur_variants(kernels: List[int]) -> List[Optional[int]]:
    """
    The images made from each image for blur_kernels, in order: None = the original image,
    otherwise the Gaussian kernel size (made odd, as GaussianBlur needs). 1 = the original.
    Empty list = no blur: only the original. A size given twice gives one image.
    """
    if not kernels:
        return [None]
    variants = []
    for kernel in kernels:
        if kernel < 1:
            raise ValueError(f"blur_kernels sizes must be >= 1 (1 = the original image): {kernels}")
        value = None if kernel == 1 else (kernel if kernel % 2 == 1 else kernel + 1)
        if value not in variants:
            variants.append(value)
    return variants


def _blur_stamp(kernels: List[int]) -> dict:
    """
    blur_kernels for the <hash> and the dataset.stamp. Before 1 existed, the original image was
    always kept: 1 is left out, and "blur_keep_original": False is added only when the original is
    dropped, so [1, 3] keeps the <hash> of the former [3] (same images).
    """
    fields = {"blur_kernels": [kernel for kernel in kernels if kernel != 1]}
    if None not in _blur_variants(kernels):
        fields["blur_keep_original"] = False
    return fields


def _add_blurred_copies(samples: List[Sample], kernels: List[int]) -> List[Sample]:
    """
    Return one copy of every sample per blur_kernels size (1 = the sample itself, not blurred).
    Empty list = the samples unchanged.
    """
    result = []
    for kernel in _blur_variants(kernels):
        if kernel is None:
            result += samples
        else:
            result += [Sample(cv2.GaussianBlur(sample.crop_bgr, (kernel, kernel), 0), sample.label,
                              sample.key, sample.image_path) for sample in samples]
    return result


def train_model(definition: TrainingDefinition, samples: List[Sample]) -> Path:
    classes = sorted({sample.label for sample in samples})
    class_to_index = {name: i for i, name in enumerate(classes)}
    print(f"[{definition.name}] {len(samples)} samples, {len(classes)} classes: "
          f"{dict(Counter(sample.label for sample in samples))}")

    train_samples, val_samples = _split(samples, definition.val_ratio)
    # Copies are added after the split, so a sample and its copies are always on the same side:
    # the validation never contains a shifted or blurred version of a training image.
    # Shift first, then blur: the shifted copies also get their blurred copies.
    if definition.shift_pixels:
        train_samples = _add_shifted_copies(train_samples, definition.shift_pixels)
        val_samples = _add_shifted_copies(val_samples, definition.shift_pixels)
        print(f"[{definition.name}] With shifted copies ({definition.shift_pixels} px, 8 directions each): "
              f"{len(train_samples)} train, {len(val_samples)} val samples.")
    if definition.blur_kernels:
        train_samples = _add_blurred_copies(train_samples, definition.blur_kernels)
        val_samples = _add_blurred_copies(val_samples, definition.blur_kernels)
        print(f"[{definition.name}] With blurred copies (kernels {definition.blur_kernels}): "
              f"{len(train_samples)} train, {len(val_samples)} val samples.")
    train_loader = DataLoader(CropDataset(train_samples, class_to_index, definition, augment=True),
                              batch_size=definition.batch_size, shuffle=True)
    # Without a validation split, the accuracy is measured on the (non augmented) training crops.
    eval_loader = DataLoader(CropDataset(val_samples or train_samples, class_to_index, definition, augment=False),
                             batch_size=definition.batch_size)
    eval_name = "val" if val_samples else "train (no val split)"

    device = _get_device()
    model = SimpleClassifier(len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=definition.learning_rate)
    criterion = nn.CrossEntropyLoss()

    acc_name = "val_acc" if val_samples else "train_acc"
    for epoch in range(definition.epochs):
        model.train()
        total_loss, total = 0.0, 0
        # One progress bar per epoch: one step per batch, with the running loss.
        with tqdm.tqdm(total=len(train_loader), unit="batch",
                       desc=f"[{definition.name}] Epoch {epoch + 1}/{definition.epochs}") as progress:
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * targets.numel()
                total += targets.numel()
                progress.set_postfix(loss=f"{total_loss / total:.4f}")
                progress.update(1)
            # The accuracy is added at the end of the bar.
            accuracy = _evaluate(model, eval_loader, device)
            progress.set_postfix({"loss": f"{total_loss / max(total, 1):.4f}", acc_name: f"{accuracy:.3f}"})

    output_dir = _output_dir(definition)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "classes": classes,
        "label": definition.label,
        "sub_categories": definition.sub_categories,
        "rect": tuple(definition.rect),
        "input_size": definition.input_size,
    }, model_path)
    with open(output_dir / "samples.json", "w", encoding="utf-8") as fp:
        json.dump([{"key": list(s.key), "image": s.image_path} for s in samples], fp, indent=4)
    print(f"[{definition.name}] Model saved: {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# 4. Process the trainings
# ---------------------------------------------------------------------------

def process_training(definition: TrainingDefinition) -> Optional[Path]:
    print(f"===== Training '{definition.name}' "
          f"(label={definition.label}, sub_categories={definition.sub_categories}, rect={definition.rect})")
    if definition.name == CROPS_PNG_DIR:
        raise ValueError(f"A TrainingDefinition can't be named '{CROPS_PNG_DIR}': "
                         f"runs/classify/<label>/{CROPS_PNG_DIR}/ is the PNG crops folder.")
    output_dir = _output_dir(definition)
    print(f"[{definition.name}] Training folder: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / TRAINING_FILE, "w", encoding="utf-8") as fp:
        json.dump({"name": definition.name, **_training_fields(definition)}, fp, indent=4)
    keys = load_or_collect_keys(definition)
    print(f"[{definition.name}] {len(keys)} unique keys found.")
    download_images(definition, keys)
    samples = load_or_build_samples(definition, keys)
    if len({sample.label for sample in samples}) < 2:
        print(f"[{definition.name}] Need at least 2 different labels to train, skipped.")
        return None
    return train_model(definition, samples)


# ---------------------------------------------------------------------------
# 5. YOLO detection (TrainingYoloDetect)
# ---------------------------------------------------------------------------

DATASET_DIR = "dataset"                 # YOLO dataset, in <hash>/
DATASET_STAMP_FILE = "dataset.stamp"    # In <hash>/dataset/
CLASSES_FILE = "classes.json"           # In <hash>/dataset/: class index -> label value
CHECK_DIR = "check"                     # Windows with their box (save_crops), in <hash>/
NOTHING_DIR = "_nothing"                # Check images of the "nothing" windows, in <hash>/check/
DETECTOR_INFO_FILE = "detector.json"    # Next to model.pt: what the inference needs
DATASET_SEED = 0                        # Same settings -> same dataset


def _window_layout(rect: Rect, jitter: float) -> Tuple[Tuple[int, int, int, int], int, int, int, int]:
    """
    :return: (box, margin_x, margin_y, width, height): the rectangle box (x0, y0, x1, y1) in card
             coordinates (rounded as crop_rect does), the margin added on each side, and the size
             of the search window. The centered window starts at (box x0 - margin_x, box y0 - margin_y),
             which is exactly crop_rect(card, rect, margin_x, margin_y).
    """
    margin_x, margin_y = _crop_margin(rect, jitter)
    x0 = int(round(rect.center_x - rect.width / 2))
    y0 = int(round(rect.center_y - rect.height / 2))
    x1, y1 = x0 + int(round(rect.width)), y0 + int(round(rect.height))
    return (x0, y0, x1, y1), margin_x, margin_y, x1 - x0 + 2 * margin_x, y1 - y0 + 2 * margin_y


def _crop_window(card_bgr: np.ndarray, x0: int, y0: int, width: int, height: int) -> np.ndarray:
    """Crop width x height pixels at (x0, y0) of a CARD_WIDTH x CARD_HEIGHT card. Pixels outside repeat the border."""
    pad = max(0, -x0, -y0, x0 + width - CARD_WIDTH, y0 + height - CARD_HEIGHT)
    if pad:
        card_bgr = cv2.copyMakeBorder(card_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return card_bgr[y0 + pad:y0 + pad + height, x0 + pad:x0 + pad + width].copy()


def _background_corner(rng: random.Random, box: Tuple[int, int, int, int], width: int,
                       height: int) -> Optional[Tuple[int, int]]:
    """
    Top-left corner of a window inside the card that doesn't touch the symbol box. Half of the time
    on the same rows as the box (ex: the rest of the type line, the hardest case), otherwise
    anywhere on the card. A wide window may not fit on the same rows: then anywhere on the card.
    None if no place was found.
    """
    bx0, by0, bx1, by1 = box
    same_rows = rng.random() < 0.5
    for attempt in range(100):
        x = rng.randint(0, CARD_WIDTH - width)
        if same_rows and attempt < 50:
            y = rng.randint(max(0, by1 - height), min(CARD_HEIGHT - height, by0))
        else:
            y = rng.randint(0, CARD_HEIGHT - height)
        if x + width <= bx0 or x >= bx1 or y + height <= by0 or y >= by1:
            return x, y
    return None


@dataclass
class DetectImage:
    window_bgr: np.ndarray
    label: Optional[str]                        # None = no symbol ("nothing")
    box: Optional[Tuple[int, int, int, int]]    # Symbol (x0, y0, x1, y1) in the window, None if nothing
    key: Tuple[str, ...]
    image_path: str


def build_detect_images(definition: TrainingYoloDetect, keys: Dict[Tuple[str, ...], List[ImageRef]],
                        rng: random.Random) -> Tuple[List[DetectImage], int]:
    """
    One window per card image, moved at random so the symbol is anywhere in it (fully visible),
    + background_ratio "nothing" windows cut from the same cards where there is no symbol.
    A card whose label value is empty (ex: a land has no mana cost '') has no symbol: its window is
    a "nothing" image.
    :return: the images and the number of card images missing or unreadable.
    """
    box, margin_x, margin_y, width, height = _window_layout(definition.rect, definition.jitter)
    bx0, by0, bx1, by1 = box
    images: List[DetectImage] = []
    missing = 0
    cut_values = Counter()  # symbol_box: label values whose box goes out of the window
    pending_backgrounds = 0.0
    for index, (key, refs) in enumerate(sorted(keys.items())):
        print(f"[{definition.name}] Key {index + 1}/{len(keys)}: {key}")
        for ref in refs:
            path = _image_path(ref)
            card_bgr = cv2.imread(str(path)) if path.exists() else None
            if card_bgr is None:
                print(f"  Image not available: {path}")
                missing += 1
                continue
            card_bgr = cv2.resize(card_bgr, (CARD_WIDTH, CARD_HEIGHT), interpolation=cv2.INTER_LINEAR)

            x = bx0 - margin_x + rng.randint(-margin_x, margin_x)
            y = by0 - margin_y + rng.randint(-margin_y, margin_y)
            if key[0]:
                if definition.symbol_box is None:
                    sx0, sy0, sx1, sy1 = box
                else:
                    sx0, sy0, sx1, sy1 = symbol_box_in_card(definition.symbol_box, key[0])
                in_window = (max(0, sx0 - x), max(0, sy0 - y), min(width, sx1 - x), min(height, sy1 - y))
                if in_window != (sx0 - x, sy0 - y, sx1 - x, sy1 - y):
                    cut_values[key[0]] += 1
                if in_window[2] <= in_window[0] or in_window[3] <= in_window[1]:
                    raise ValueError(f"[{definition.name}] The box of {key[0]!r} is out of the window: "
                                     f"make the rect contain it")
                images.append(DetectImage(_crop_window(card_bgr, x, y, width, height), key[0],
                                          in_window, key, str(path)))
            else:
                images.append(DetectImage(_crop_window(card_bgr, x, y, width, height), None, None, key, str(path)))

            pending_backgrounds += definition.background_ratio
            while pending_backgrounds >= 1.0:
                pending_backgrounds -= 1.0
                corner = _background_corner(rng, box, width, height)
                if corner is not None:
                    images.append(DetectImage(_crop_window(card_bgr, *corner, width, height),
                                              None, None, key, str(path)))
    if cut_values:
        print(f"[{definition.name}] WARNING: {sum(cut_values.values())} boxes go out of the window and are "
              f"cut at its border (make the rect wider to keep them whole): {dict(cut_values)}")
    return images, missing


def _split_detect(images: List[DetectImage], val_ratio: float,
                  rng: random.Random) -> Tuple[List[DetectImage], List[DetectImage]]:
    """Split per label ("nothing" is its own group) so that every label keeps at least one training image."""
    by_label: Dict[Optional[str], List[DetectImage]] = defaultdict(list)
    for image in images:
        by_label[image.label].append(image)
    train, val = [], []
    for label in sorted(by_label, key=lambda value: (value is None, value or "")):
        group = by_label[label]
        rng.shuffle(group)
        n_val = int(len(group) * val_ratio)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def _downscale_copy(window_bgr: np.ndarray, factor: float, rng: random.Random) -> np.ndarray:
    """
    The window as the webcam sees a small card: shrunk by `factor` (the sensor averages the pixels),
    a little sensor noise, JPEG compression (quality 40-80), then enlarged back to its size as
    reshape_card does. The symbol box doesn't move.
    """
    height, width = window_bgr.shape[:2]
    small = cv2.resize(window_bgr, (max(1, round(width * factor)), max(1, round(height * factor))),
                       interpolation=cv2.INTER_AREA)
    noise = np.random.default_rng(rng.randrange(2 ** 32)).normal(0.0, 3.0, small.shape)
    small = np.clip(small + noise, 0, 255).astype(np.uint8)
    _, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 80)])
    small = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _write_yolo_split(images: List[DetectImage], class_to_index: Dict[str, int], dataset_dir: Path,
                      split: str, blur_kernels: List[int], downscale: List[float],
                      rng: random.Random) -> List[dict]:
    """
    Write images/<split>/*.png and labels/<split>/*.txt (YOLO format, empty file = nothing): one
    image per blur_kernels size (1 = the image itself, see _blur_variants) and one low resolution
    copy of the original image per downscale factor.
    The copies stay in the same split as their image.
    :return: one entry per written image, for images.json.
    """
    image_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    # (kind, value): the image itself, then its copies.
    variants = ([("original", None) if kernel is None else ("blur", kernel) for kernel in _blur_variants(blur_kernels)]
                + [("downscale", factor) for factor in downscale])
    entries = []
    index = 0
    for image in tqdm.tqdm(images, desc=f"Write {split}", unit="image"):
        if image.label is None:
            line = ""
        else:
            height, width = image.window_bgr.shape[:2]
            x0, y0, x1, y1 = image.box
            line = (f"{class_to_index[image.label]} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                    f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}\n")
        for kind, value in variants:
            if kind == "blur":
                window = cv2.GaussianBlur(image.window_bgr, (value, value), 0)
            elif kind == "downscale":
                window = _downscale_copy(image.window_bgr, value, rng)
            else:
                window = image.window_bgr
            name = f"{index:06d}"
            cv2.imwrite(str(image_dir / f"{name}.png"), window)
            with open(label_dir / f"{name}.txt", "w", encoding="utf-8") as fp:
                fp.write(line)
            entries.append({"file": f"images/{split}/{name}.png", "label": image.label, "box": image.box,
                            "blur": value if kind == "blur" else None,
                            "downscale": value if kind == "downscale" else None,
                            "key": list(image.key), "image": image.image_path})
            index += 1
    return entries


def _save_check_images(definition: TrainingYoloDetect, images: List[DetectImage]):
    """<hash>/check/<label>/<sub_categories>_<id>.png: each window with its box, to check the rect."""
    check_dir = _output_dir(definition) / CHECK_DIR
    if check_dir.exists():
        shutil.rmtree(check_dir)
    for image in images:
        window = image.window_bgr.copy()
        if image.label is None:
            folder = check_dir / NOTHING_DIR
        else:
            folder = check_dir / _safe(image.label)
            x0, y0, x1, y1 = image.box
            cv2.rectangle(window, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
        folder.mkdir(parents=True, exist_ok=True)
        name = "_".join(_safe(v) for v in image.key[1:]) or "key"
        cv2.imwrite(str(folder / f"{name}_{Path(image.image_path).stem[:8]}.png"), window)


def _dataset_stamp(definition: TrainingYoloDetect, keys: Dict[Tuple[str, ...], List[ImageRef]]) -> dict:
    """What the YOLO dataset depends on."""
    return {
        "keys": _keys_stamp(definition),
        "keys_hash": _keys_hash(keys),
        "rect": list(definition.rect),
        "jitter": definition.jitter,
        **_blur_stamp(definition.blur_kernels),
        "val_ratio": definition.val_ratio,
        "background_ratio": definition.background_ratio,
        "seed": DATASET_SEED,
        **({"downscale": list(definition.downscale)} if definition.downscale else {}),
        **({"symbol_box": list(definition.symbol_box)} if definition.symbol_box is not None else {}),
    }


def load_or_build_detect_dataset(definition: TrainingYoloDetect,
                                 keys: Dict[Tuple[str, ...], List[ImageRef]]) -> Optional[Tuple[Path, List[str]]]:
    """
    Build <hash>/dataset/ (YOLO format: data.yaml, images/, labels/, classes.json, images.json),
    unless dataset.stamp matches the current TrainingYoloDetect and keys.
    :return: (data.yaml path, classes), or None if there is no symbol image.
    """
    dataset_dir = _output_dir(definition) / DATASET_DIR
    yaml_path = dataset_dir / "data.yaml"
    classes_path = dataset_dir / CLASSES_FILE
    stamp_path = dataset_dir / DATASET_STAMP_FILE
    stamp = _dataset_stamp(definition, keys)

    if yaml_path.exists() and classes_path.exists() and _stamp_matches(stamp_path, stamp):
        with open(classes_path, "r", encoding="utf-8") as fp:
            classes = json.load(fp)
        print(f"[{definition.name}] YOLO dataset loaded from {dataset_dir} ({DATASET_STAMP_FILE} is up to date).")
        return yaml_path, classes

    if any(not 0 < factor < 1 for factor in definition.downscale):
        raise ValueError(f"[{definition.name}] downscale factors must be between 0 and 1: {definition.downscale}")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)  # Never mix images of two different settings
    rng = random.Random(DATASET_SEED)
    images, missing = build_detect_images(definition, keys, rng)
    classes = sorted({image.label for image in images if image.label is not None})
    if not classes:
        return None
    class_to_index = {name: i for i, name in enumerate(classes)}
    n_nothing = sum(image.label is None for image in images)
    print(f"[{definition.name}] {len(images) - n_nothing} symbol images ({len(classes)} classes), "
          f"{n_nothing} nothing images: {dict(Counter(image.label or '<nothing>' for image in images))}")
    if definition.save_crops:
        _save_check_images(definition, images)

    train, val = _split_detect(images, definition.val_ratio, rng)
    entries = _write_yolo_split(train, class_to_index, dataset_dir, "train", definition.blur_kernels,
                                definition.downscale, rng)
    if val:
        entries += _write_yolo_split(val, class_to_index, dataset_dir, "val", definition.blur_kernels,
                                     definition.downscale, rng)
    else:
        print(f"[{definition.name}] No validation split: the validation uses the training images.")
    print(f"[{definition.name}] YOLO dataset: {sum(e['file'].startswith('images/train') for e in entries)} train, "
          f"{sum(e['file'].startswith('images/val') for e in entries)} val images (with blurred / downscaled copies).")

    with open(yaml_path, "w", encoding="utf-8") as fp:
        fp.write(f"path: {json.dumps(dataset_dir.resolve().as_posix())}\n"
                 f"train: images/train\n"
                 f"val: images/{'val' if val else 'train'}\n"
                 f"nc: {len(classes)}\n"
                 f"names:\n" + "".join(f"  {i}: {json.dumps(name)}\n" for i, name in enumerate(classes)))
    with open(classes_path, "w", encoding="utf-8") as fp:
        json.dump(classes, fp, indent=4)
    with open(dataset_dir / "images.json", "w", encoding="utf-8") as fp:
        json.dump(entries, fp, indent=4)
    if missing:
        # Some images were missing or unreadable: no stamp, the dataset is built again next time.
        print(f"[{definition.name}] {missing} image(s) missing, {DATASET_STAMP_FILE} not written.")
    else:
        _write_stamp(stamp_path, stamp)
    return yaml_path, classes


def train_yolo_detect(definition: TrainingYoloDetect, yaml_path: Path, classes: List[str]) -> Path:
    """Train the YOLO detector. Saved as <hash>/model.pt + <hash>/detector.json (the Ultralytics run is in <hash>/yolo/)."""
    from ultralytics import YOLO  # Only needed to train or to detect

    output_dir = _output_dir(definition)
    local_weights = ROOT / definition.model
    model = YOLO(str(local_weights) if local_weights.exists() else definition.model)
    model.train(
        data=str(yaml_path),
        imgsz=definition.input_size,
        epochs=definition.epochs,
        batch=definition.batch_size,
        optimizer="AdamW",              # With optimizer="auto", Ultralytics ignores lr0
        lr0=definition.learning_rate,
        project=str(output_dir),        # Absolute path: the run is in <hash>/yolo/
        name="yolo",
        exist_ok=True,
        workers=definition.workers,
        patience=definition.patience,
        cache=definition.cache,
        # The window comes from a straightened card: the symbol is never mirrored.
        fliplr=0.0,
        translate=definition.translate,
        scale=definition.scale,
        mosaic=definition.mosaic,
    )

    best_path = output_dir / "yolo" / "weights" / "best.pt"
    model_path = output_dir / "model.pt"
    shutil.copy2(best_path, model_path)
    with open(output_dir / DETECTOR_INFO_FILE, "w", encoding="utf-8") as fp:
        json.dump({
            "label": definition.label,
            "sub_categories": list(definition.sub_categories),
            "classes": classes,
            "rect": list(definition.rect),
            "jitter": definition.jitter,
            "input_size": definition.input_size,
        }, fp, indent=4)
    print(f"[{definition.name}] Model saved: {model_path}")
    return model_path


def process_yolo_detect_training(definition: TrainingYoloDetect) -> Optional[Path]:
    print(f"===== YOLO detect training '{definition.name}' "
          f"(label={definition.label}, sub_categories={definition.sub_categories}, rect={definition.rect})")
    output_dir = _output_dir(definition)
    print(f"[{definition.name}] Training folder: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / TRAINING_FILE, "w", encoding="utf-8") as fp:
        json.dump({"name": definition.name, **_training_fields(definition)}, fp, indent=4)
    keys = load_or_collect_keys(definition)
    print(f"[{definition.name}] {len(keys)} unique keys found.")
    download_images(definition, keys)
    dataset = load_or_build_detect_dataset(definition, keys)
    if dataset is None:
        print(f"[{definition.name}] No symbol image, skipped.")
        return None
    yaml_path, classes = dataset
    return train_yolo_detect(definition, yaml_path, classes)


def training():
    # Filter possibilities :
    # - "frame": ["2003", "2015"]
    # Same cards for every 2015-frame training: normal layout, planeswalkers excluded.
    accept_2015 = {"frame": ["2015"], "layout": ["normal"]}
    filters_2015 = {"type_line": ["Legendary Planeswalker", "Planeswalker"]}
    definitions = [
        # Expansion with a YOLO detector (runs/detect/set/set/): "no symbol" or the expansion code.
        # downscale: the symbol as seen by the webcam (about 35-60% of the scryfall resolution).
        TrainingYoloDetect(
            name="set",
            label="set",
            sub_categories=["rarity", "colors"],
            rect=Rect(center_x=430, center_y=402, height=64, width=64),
            accept=accept_2015,
            filters=filters_2015,
            samples_per_key=1,
            epochs=25,
            batch_size=32,
            blur_kernels=[1, 3],
            downscale=[0.35, 0.45, 0.6],
            cache="ram",  # About 0.8 GB: no slow /mnt/c read after the first epoch
        ),
        # Mana cost with a YOLO detector (runs/detect/mana_cost/mana_cost/): "no mana cost" or the cost,
        # ex: '{1}{W}{W}'. Read at the top right of the card. Lands (no mana cost) are "nothing" images.
        # The rect is the search window (jitter 0: same window for every card). The box of each image
        # only covers its symbols: n symbols -> n x 25.3 px wide, 28 px high, right edge 35 px from
        # the right border, top 37 px from the top.
        TrainingYoloDetect(
            name="mana_cost",
            label="mana_cost",
            sub_categories=[],
            rect=Rect(center_x=334, center_y=51, height=102, width=308),
            symbol_box=SymbolBox(symbol_width=253 / 10, height=28, right=35, top=37),
            accept={"frame": ["2003", "2015"], "layout": ["normal", "flip", "transform", "modal_dfc", "leveler"]},
            filters=filters_2015,
            samples_per_key=10,
            input_size=320,
            jitter=0.0,
            epochs=25,
            batch_size=32,
            blur_kernels=[5, 7, 9],
            cache="ram",  # About 0.8 GB: no slow /mnt/c read after the first epoch
            translate=0.05,
            mosaic=1.0,
        ),
        # Expansion, read on the same symbol, with the basic CNN (runs/classify/set/set/).
        # The symbol color depends on the rarity.
        TrainingDefinition(
            name="set",
            label="set",
            sub_categories=["rarity", "colors"],
            rect=Rect(center_x=430, center_y=402, height=64, width=64),
            accept=accept_2015,
            filters=filters_2015,
            samples_per_key=1,
            epochs=20,
            batch_size=32,
            blur_kernels=[1, 3],
            shift_pixels=[2, 4, 6, 8, 10, 12, 14 ,16],
        ),
        # Rarity, read on the expansion symbol, with the basic CNN. The symbol depends on the set.
        TrainingDefinition(
            name="rarity",
            label="rarity",
            sub_categories=["set"],
            rect=Rect(center_x=430, center_y=400, height=80, width=80),
            accept={"frame": ["2003", "2015"]},
        ),
    ]

    for definition in _choose_trainings(definitions):
        if isinstance(definition, TrainingYoloDetect):
            process_yolo_detect_training(definition)
        else:
            process_training(definition)


def _choose_trainings(definitions: list) -> list:
    """Ask which training(s) to run: a number, several numbers (1,3) or * for all."""
    print("Trainings:")
    for index, definition in enumerate(definitions, start=1):
        kind = "YOLO detect" if isinstance(definition, TrainingYoloDetect) else "basic CNN"
        print(f"  {index}. {definition.name:<10} label '{definition.label}', {kind}")
    while True:
        answer = input(f"Which training ? (1-{len(definitions)}, several: 1,3, all: *) : ").strip()
        if answer == "*":
            return definitions
        try:
            numbers = [int(part) for part in answer.split(",") if part.strip()]
        except ValueError:
            numbers = []
        if numbers and all(1 <= number <= len(definitions) for number in numbers):
            return [definitions[number - 1] for number in numbers]
        print(f"'{answer}' is not a valid choice.")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_classifier(model_path) -> dict:
    bundle = torch.load(model_path, map_location="cpu")
    model = SimpleClassifier(len(bundle["classes"]))
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    bundle["model"] = model
    bundle["rect"] = Rect(*bundle["rect"])
    return bundle


def predict(bundle: dict, card_bgr: np.ndarray) -> Tuple[str, float]:
    """Classify a straightened card image (any size, it is resized to CARD_WIDTH x CARD_HEIGHT)."""
    crop = crop_rect(card_bgr, bundle["rect"])
    with torch.no_grad():
        probabilities = torch.softmax(bundle["model"](_to_tensor(crop, bundle["input_size"])[None]), dim=1)[0]
    index = int(probabilities.argmax())
    return bundle["classes"][index], float(probabilities[index])


def load_detector(model_path) -> dict:
    """Load a TrainingYoloDetect model: <hash>/model.pt and the detector.json next to it."""
    from ultralytics import YOLO  # Only needed to train or to detect

    model_path = Path(model_path)
    with open(model_path.parent / DETECTOR_INFO_FILE, "r", encoding="utf-8") as fp:
        bundle = json.load(fp)
    bundle["rect"] = Rect(*bundle["rect"])
    bundle["model"] = YOLO(str(model_path))
    return bundle


def search_window(bundle: dict) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the search window of a detector, in CARD_WIDTH x CARD_HEIGHT card coordinates."""
    box, margin_x, margin_y, width, height = _window_layout(bundle["rect"], bundle["jitter"])
    x0, y0 = box[0] - margin_x, box[1] - margin_y
    return x0, y0, x0 + width, y0 + height


def detect(bundle: dict, card_bgr: np.ndarray, min_confidence: float = 0.25
           ) -> Tuple[Optional[str], float, Optional[Tuple[int, int, int, int]]]:
    """
    Look for the symbol in the search window of a straightened card (any size, it is resized to
    CARD_WIDTH x CARD_HEIGHT).
    :return: (label value, confidence, box) of the best box, the box (x0, y0, x1, y1) being in
             CARD_WIDTH x CARD_HEIGHT card coordinates. (None, 0.0, None) if no symbol was found.
    """
    card_bgr = cv2.resize(card_bgr, (CARD_WIDTH, CARD_HEIGHT), interpolation=cv2.INTER_LINEAR)
    x0, y0, x1, y1 = search_window(bundle)
    window = _crop_window(card_bgr, x0, y0, x1 - x0, y1 - y0)
    result = bundle["model"](window, imgsz=bundle["input_size"], conf=min_confidence, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None, 0.0, None
    best = int(result.boxes.conf.argmax())
    bx0, by0, bx1, by1 = (int(round(v)) for v in result.boxes.xyxy[best].tolist())
    return (bundle["classes"][int(result.boxes.cls[best])], float(result.boxes.conf[best]),
            (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0))


if __name__ == "__main__":
    training()
