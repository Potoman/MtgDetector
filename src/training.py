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
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

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
    # Gaussian blur kernel sizes (odd numbers). Each size adds one blurred copy of every sample:
    # [5] doubles the dataset, [3, 7] triples it. Empty = no blurred copies.
    blur_kernels: List[int] = field(default_factory=list)
    # Shifted copies: for each distance of the list, every sample is moved by that many pixels
    # (card coordinates) in the 8 directions (straight and diagonal). The uncovered side repeats
    # the border pixels. Each distance gives 8 more samples per sample:
    # [4] -> dataset x9, [3, 6, 9] -> dataset x25. Empty = no shifted copies.
    shift_pixels: List[int] = field(default_factory=list)


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


def _label_dir(definition: TrainingDefinition) -> Path:
    """runs/classify/<label>/"""
    return OUTPUT_DIR / _safe(definition.label)


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
    Not used: name (already in the path) and save_crops (only writes the PNG check images).
    """
    return {
        "label": definition.label,
        "sub_categories": list(definition.sub_categories),
        "exp_codes": definition.exp_codes,
        "accept": {name: _as_list(values) for name, values in definition.accept.items()},
        "filters": {name: _as_list(values) for name, values in definition.filters.items()},
        "samples_per_key": definition.samples_per_key,
        "epochs": definition.epochs,
        "batch_size": definition.batch_size,
        "blur_kernels": list(definition.blur_kernels),
        "shift_pixels": definition.shift_pixels if isinstance(definition.shift_pixels, int) else list(definition.shift_pixels),
        "rect": list(definition.rect),
        "jitter": definition.jitter,
        "input_size": definition.input_size,
        "learning_rate": definition.learning_rate,
        "val_ratio": definition.val_ratio,
    }


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
    """
    output_dir = _output_dir(definition)
    for other_stamp in sorted(_label_dir(definition).glob(f"*/*/{stamp_name}")):
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


def _add_blurred_copies(samples: List[Sample], kernels: List[int]) -> List[Sample]:
    """Return the samples + one blurred copy of every sample for each kernel size."""
    copies = []
    for kernel in kernels:
        kernel = kernel if kernel % 2 == 1 else kernel + 1  # GaussianBlur needs an odd size
        for sample in samples:
            blurred = cv2.GaussianBlur(sample.crop_bgr, (kernel, kernel), 0)
            copies.append(Sample(blurred, sample.label, sample.key, sample.image_path))
    return samples + copies


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


def training():
    # Filter possibilities :
    # - "frame": ["2003", "2015"]
    definitions = [
        # Rarity, read on the expansion symbol. The symbol depends on the set.
        #TrainingDefinition(
        #    name="rarity",
        #    label="rarity",
        #    sub_categories=["set"],
        #    rect=Rect(center_x=430, center_y=400, height=80, width=80),
        #    accept={"frame": ["2003", "2015"],
        #            "set": },
        #),
        # Expansion, read on the same symbol. The symbol color depends on the rarity.
        TrainingDefinition(
            name="set",
            label="set",
            sub_categories=["rarity", "colors"],
            rect=Rect(center_x=436, center_y=402, height=64, width=64),
            accept={"frame": ["2015"], "layout": ["normal"]},
            filters={"type_line": ["Legendary Planeswalker", "Planeswalker"]},
            samples_per_key=4,
            epochs=20,
            batch_size=32,
            blur_kernels=[3, 5],
            shift_pixels=[2, 4, 6],
        ),
    ]

    for definition in definitions:
        process_training(definition)


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


if __name__ == "__main__":
    training()
