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
3. The images of each key are downloaded if needed (download.download_if_not_exist) and the
   rectangle is cropped. The crop is bound to the label value.
4. A small PyTorch CNN classifier is trained on the crops and saved in
   <root>/runs/classify/<name>/model.pt

Uses PyTorch only. Do not import TensorFlow (or modules that import it, such as
algo_corner_detection, dataset or algo) from this file: loading TensorFlow and PyTorch in
the same process can make their CUDA libraries clash.

Run from anywhere: python3 src/training.py
"""
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import download


ROOT = download.ROOT
SCRYFALL_DIR = download.SCRYFALL_DIR
OUTPUT_DIR = ROOT / "runs" / "classify"

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
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-3
    val_ratio: float = 0.2          # Only applied to labels having at least 2 samples
    save_crops: bool = True         # Save the crops in runs/classify/<name>/crops to check the rectangle


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


# ---------------------------------------------------------------------------
# 2. Get the images and crop the rectangle
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
    crops_dir = OUTPUT_DIR / definition.name / "crops"

    samples: List[Sample] = []
    for index, (key, images) in enumerate(sorted(keys.items())):
        print(f"[{definition.name}] Key {index + 1}/{len(keys)}: {key}")
        for image in images:
            image_dir = SCRYFALL_DIR / f"exp_{image.exp_code}" / "image"
            expected_path = image_dir / f"{image.illustration_id}.jpg"
            # Download all faces of the card if needed, then keep the face of this key.
            paths = download.download_if_not_exist(image.exp_code, image.card_id)
            if str(expected_path) not in paths:
                print(f"  Image not available: {expected_path}")
                continue

            card_bgr = cv2.imread(str(expected_path))
            if card_bgr is None:
                print(f"  Unreadable image: {expected_path}")
                continue

            crop = crop_rect(card_bgr, definition.rect, margin_x, margin_y)
            samples.append(Sample(crop, key[0], key, str(expected_path)))

            if definition.save_crops:
                label_dir = crops_dir / _safe(key[0])
                label_dir.mkdir(parents=True, exist_ok=True)
                name = "_".join(_safe(v) for v in key[1:]) or "key"
                cv2.imwrite(str(label_dir / f"{name}_{image.illustration_id[:8]}.png"),
                            crop[margin_y:crop.shape[0] - margin_y, margin_x:crop.shape[1] - margin_x])
    return samples


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


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


def train_model(definition: TrainingDefinition, samples: List[Sample]) -> Path:
    classes = sorted({sample.label for sample in samples})
    class_to_index = {name: i for i, name in enumerate(classes)}
    print(f"[{definition.name}] {len(samples)} samples, {len(classes)} classes: "
          f"{dict(Counter(sample.label for sample in samples))}")

    train_samples, val_samples = _split(samples, definition.val_ratio)
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

    for epoch in range(definition.epochs):
        model.train()
        total_loss, total = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            total += targets.numel()
        accuracy = _evaluate(model, eval_loader, device)
        print(f"[{definition.name}] Epoch {epoch + 1}/{definition.epochs} "
              f"loss={total_loss / max(total, 1):.4f} {eval_name} acc={accuracy:.3f}")

    output_dir = OUTPUT_DIR / definition.name
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
    keys = collect_keys(definition)
    print(f"[{definition.name}] {len(keys)} unique keys found.")
    samples = build_samples(definition, keys)
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
            rect=Rect(center_x=420, center_y=400, height=32, width=70),
            accept={"frame": ["2015"], "layout": ["normal"]},
            filters={"type_line": ["Legendary Planeswalker", "Planeswalker"]}
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
