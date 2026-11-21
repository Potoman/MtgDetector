"""
Power / toughness classification: a YOLO classifier (Ultralytics, task classify) that reads the power /
toughness printed at the bottom right of a straightened card, in the window power_toughness_data_set.VAL_RECT
(100 x 60, x 364-464, y 592-652 of a CARD_WIDTH x CARD_HEIGHT card). One class per value ('2/3', '*/1+*'...,
the values of the local scryfall data), and 'none' (power_toughness_data_set.NO_VALUE): a card without power /
toughness. No box: the image is the window.

'pt_gen' (PowerToughnessTraining): trained on the generated images of power_toughness_data_set.py and
validated on its real cards: runs/classify/power_toughness/gen/<n>_val<percent>_nopt<percent>/, n, the percent
of validation and the percent without power / toughness asked at launch, built only if its dataset.stamp
doesn't match.
Input: the window resized to input_width x input_height (160 x 96: not square, so the text is not cut).
The Ultralytics classification transforms (a random square crop to train, a center square crop to validate)
are replaced (power_toughness_trainer). The augmentations of the training images, redone at every epoch, are
the ones of the set symbol detector (training_set_symbol.py), in the Ultralytics order: zoom +/- scale (20%,
the border repeated), the albumentations (blur, gamma), then the color (hsv_s, hsv_v, no hue change). No flip.
The real cards of the validation get the same color and blur changes, once, when the dataset is built.
Validation: metrics/accuracy_top1, the share of real cards read right, is the fitness (chooses best.pt and
stops the training). Also printed: the share read right of each frame (1993, 1997, 2003, 2015, future) and how
the cards without power / toughness are read. The wrong reads of the last validation (best.pt, after the training)
are saved in <hash>/yolo/pt_errors.json, the failures of each class in <hash>/yolo/class_errors.csv.
Files: runs/classify/power_toughness/<name>/<hash>/: training.json (the settings: <hash> is computed on
them), yolo/ (the Ultralytics run: results.csv, results.png, confusion matrix...), model.pt, classifier.json
(what the inference needs, see load_classifier / read_power_toughness).
Run from anywhere: python3 src/training_power_toughness.py (menu of these trainings)
"""
import csv
import dataclasses
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import local_paths
import power_toughness_data_set as data
import set_symbol_data_set
import training

OUTPUT_DIR = local_paths.RUNS_DIR / "classify" / "power_toughness"
TRAINING_FILE = "training.json"
CLASSIFIER_INFO_FILE = "classifier.json"    # Next to model.pt: what the inference needs
PT_ERRORS_FILE = "pt_errors.json"           # In the Ultralytics run folder (<hash>/yolo/)
CLASS_ERRORS_FILE = "class_errors.csv"      # Same folder: the failures of each class
WORST_CLASSES_PRINTED = 10
READ_AS_KEPT = 5


@dataclass
class PowerToughnessTraining:
    """The power / toughness classifier, trained on the dataset of power_toughness_data_set.py."""
    name: str                               # Output folder: runs/classify/power_toughness/<name>
    label: str = "power_toughness"          # Shown in the menu
    samples: int = 0                        # n of power_toughness_data_set.py. 0 = asked at launch
    val_percent: float = 0.0                # Percent of the samples for the validation (real cards). 0 = asked
    no_pt_percent: float = -1.0             # Percent of each split without power / toughness. -1 = asked
    model: str = "yolo26s-cls.pt"           # Pretrained classifier: <root>/<model> if it exists, else downloaded
    input_width: int = 160                  # Model input (multiples of 32): the 100 x 60 window is resized to it
    input_height: int = 96
    epochs: int = 40
    patience: int = 10                      # Stop when accuracy_top1 didn't improve for this many epochs
    batch_size: int = 256
    learning_rate: float = 1e-3             # Initial learning rate of the AdamW optimizer (YOLO lr0)
    workers: int = 8                        # Processes preparing the images (only the speed)
    cache: Union[bool, str] = "ram"         # Ultralytics image cache (only the speed)
    # Augmentations of the training images, redone at every epoch (see the top of the file).
    scale: float = 0.2                      # Zoom +/-
    hsv_h: float = 0.0
    hsv_s: float = 0.4
    hsv_v: float = 0.3
    augmentations: List[Tuple[str, dict]] = field(default_factory=list)  # Albumentations: (name, parameters)

    KIND = "YOLO classify, power / toughness, generated images"

    def run(self) -> Optional[Path]:
        return process_power_toughness_training(self)


def definitions() -> List[PowerToughnessTraining]:
    """The power / toughness trainings, in the order of the menu."""
    return [
        PowerToughnessTraining(
            name="pt_gen",
            # As set_gen (training_set_symbol.py): a blur on half of the images (the webcam is blurry), gamma,
            # lighter color changes than the Ultralytics defaults, no hue change.
            augmentations=[
                ("Blur", {"blur_limit": (3), "p": 0.5}),
                ("RandomGamma", {"gamma_limit": (80, 120), "p": 0.3}),
                ("ToGray", {"p": 0.0}),
            ],
        ),
    ]


def val_augmentation(definition: PowerToughnessTraining) -> dict:
    """The augmentation of the real cards of the validation: the color and blur changes of the training."""
    return set_symbol_data_set.val_augmentation_of(definition.hsv_h, definition.hsv_s, definition.hsv_v,
                                                   definition.augmentations)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def training_fields(definition: PowerToughnessTraining) -> dict:
    """The settings that change the trained model (<hash>): all but name, workers and cache."""
    fields = dataclasses.asdict(definition)
    for name in ("name", "workers", "cache"):
        fields.pop(name)
    fields["augmentations"] = [[name, dict(params)] for name, params in definition.augmentations]
    fields["dataset_version"] = data.VERSION
    fields["val_rect"] = list(data.VAL_RECT)
    return json.loads(json.dumps(fields))


def output_dir(definition: PowerToughnessTraining) -> Path:
    """runs/classify/power_toughness/<name>/<hash>/: training.json, yolo/, model.pt, classifier.json."""
    content = json.dumps(training_fields(definition), sort_keys=True)
    return OUTPUT_DIR / definition.name / hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]


def process_power_toughness_training(definition: PowerToughnessTraining) -> Optional[Path]:
    """Ask n and the percents if needed, load or build the dataset of power_toughness_data_set.py, then train."""
    if definition.samples <= 0:
        definition = dataclasses.replace(definition, samples=data.ask_samples())
    if definition.val_percent <= 0:
        definition = dataclasses.replace(definition, val_percent=data.ask_val_percent(definition.samples))
    if definition.no_pt_percent < 0:
        definition = dataclasses.replace(definition, no_pt_percent=data.ask_no_pt_percent(definition.samples,
                                                                                          definition.val_percent))
    folder = output_dir(definition)
    print(f"===== {definition.KIND} training '{definition.name}': {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / TRAINING_FILE, "w", encoding="utf-8") as fp:
        json.dump({"name": definition.name, **training_fields(definition)}, fp, indent=4)
    dataset, values = data.load_or_build_dataset(definition.samples, definition.val_percent, definition.no_pt_percent,
                                                 val_augmentation=val_augmentation(definition))
    return train(definition, dataset, values)


def _albumentations(definition: PowerToughnessTraining):
    """The albumentations Compose of the training (None without augmentations)."""
    if not definition.augmentations:
        return None
    try:
        import albumentations as A
    except ImportError as error:
        raise ImportError(f"[{definition.name}] 'augmentations' needs the albumentations package: "
                          f"pip install albumentations") from error
    return A.Compose([getattr(A, name)(**params) for name, params in definition.augmentations])


def train(definition: PowerToughnessTraining, dataset: Path, values: Dict[str, str]) -> Path:
    """
    Train the classifier on the dataset folder (one folder per class, values: folder -> value).
    Saved as <hash>/model.pt + <hash>/classifier.json (the Ultralytics run is in <hash>/yolo/).
    """
    _albumentations(definition)  # Fails now if albumentations is missing
    from ultralytics import YOLO  # Only needed to train or to classify

    folder = output_dir(definition)
    local_weights = training.ROOT / definition.model
    model = YOLO(str(local_weights) if local_weights.exists() else definition.model)
    model.add_callback("on_fit_epoch_end", training._print_epoch_losses)
    model.train(
        trainer=power_toughness_trainer(definition, values),
        data=str(dataset),
        imgsz=max(definition.input_width, definition.input_height),  # Not used by the images (see the trainer)
        epochs=definition.epochs,
        batch=definition.batch_size,
        optimizer="AdamW",              # With optimizer="auto", Ultralytics ignores lr0
        lr0=definition.learning_rate,
        project=str(folder),            # Absolute path: the run is in <hash>/yolo/
        name="yolo",
        exist_ok=True,
        workers=definition.workers,
        patience=definition.patience,
        cache=definition.cache,
        fliplr=0.0,                     # Never mirrored (a straightened card)
        scale=definition.scale,
        hsv_h=definition.hsv_h,
        hsv_s=definition.hsv_s,
        hsv_v=definition.hsv_v,
    )
    model_path = folder / "model.pt"
    shutil.copy2(folder / "yolo" / "weights" / "best.pt", model_path)
    with open(folder / CLASSIFIER_INFO_FILE, "w", encoding="utf-8") as fp:
        json.dump({"label": definition.label, "classifier": "power_toughness", "rect": list(data.VAL_RECT),
                   "input_size": [definition.input_width, definition.input_height], "values": values,
                   "dataset": str(dataset)}, fp, indent=4, ensure_ascii=False)
    print(f"[{definition.name}] Model saved: {model_path}")
    print_summary(folder / "yolo", values)
    return model_path


# ---------------------------------------------------------------------------
# Images: augmentation and model input
# ---------------------------------------------------------------------------

def to_input(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    The model input of a window: resized to width x height, RGB, CHW uint8. Divided by 255 on the GPU (as the
    Ultralytics detection): float32 batches are 4 times bigger, too big for the pinned memory of WSL.
    """
    image = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))


class TrainAugmenter:
    """
    The augmentations of a training image, in the order of the Ultralytics detection training: zoom (as
    RandomPerspective, scale), the albumentations, then RandomHSV. Random: np.random (seeded per dataloader
    worker by Ultralytics) and the albumentations, seeded again in each worker process.
    """

    def __init__(self, definition: PowerToughnessTraining):
        from ultralytics.data.augment import RandomHSV

        self.scale = definition.scale
        self.compose = _albumentations(definition)
        self.hsv = RandomHSV(hgain=definition.hsv_h, sgain=definition.hsv_s, vgain=definition.hsv_v)
        self.pid = None

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.pid != os.getpid():  # A new dataloader worker: its own albumentations draws
            self.pid = os.getpid()
            if self.compose is not None:
                self.compose.set_random_seed(int(np.random.randint(2 ** 31)))
        image = image_bgr.copy()  # Never change the cached image
        if self.scale:
            s = random.uniform(1 - self.scale, 1 + self.scale)
            height, width = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, s)
            image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        if self.compose is not None:
            image = self.compose(image=image)["image"]
        self.hsv.apply_image({"img": image})  # In place
        return image


# ---------------------------------------------------------------------------
# Ultralytics trainer and validator
# ---------------------------------------------------------------------------

def class_scores(reads) -> List[dict]:
    """
    From (image, expected, read, confidence): the failures of each expected value, the most failed first
    (percent, then images). 'read_as': the READ_AS_KEPT most frequent wrong reads.
    """
    by_class: Dict[str, dict] = {}
    for _, expected, read, _ in reads:
        entry = by_class.setdefault(expected, {"class": expected, "images": 0, "failed": 0, "failed_percent": 0.0,
                                               "read_as": Counter()})
        entry["images"] += 1
        if read != expected:
            entry["failed"] += 1
            entry["read_as"][read] += 1
    rows = []
    for entry in by_class.values():
        entry["failed_percent"] = round(100 * entry["failed"] / entry["images"], 1)
        entry["read_as"] = dict(entry["read_as"].most_common(READ_AS_KEPT))
        rows.append(entry)
    return sorted(rows, key=lambda row: (-row["failed_percent"], -row["images"], row["class"]))


IMAGES_FILE = "images.json"     # Of the dataset (power_toughness_data_set.py): the frame of each validation card


def val_frames(image_file: str) -> Dict[str, str]:
    """The frame of each validation image of the dataset of image_file (its images.json), by absolute path."""
    root = Path(image_file).resolve().parents[2]  # <dataset>/val/<folder>/<name>.png
    with open(root / IMAGES_FILE, "r", encoding="utf-8") as fp:
        entries = json.load(fp)
    return {str((root / entry["file"]).resolve()): entry["frame"] for entry in entries if entry.get("frame")}


def frame_scores(reads, frames: Dict[str, str]) -> Dict[str, dict]:
    """From (image, expected, read, confidence): the share read right of each frame ('?': unknown)."""
    scores: Dict[str, dict] = {}
    for image, expected, read, _ in reads:
        entry = scores.setdefault(frames.get(str(Path(image).resolve()), "?"), {"images": 0, "right": 0})
        entry["images"] += 1
        entry["right"] += read == expected
    for entry in scores.values():
        entry["accuracy"] = round(entry["right"] / entry["images"], 4)
    return dict(sorted(scores.items()))


def format_frame_scores(scores: Dict[str, dict]) -> str:
    return "Read right per frame: " + ", ".join(f"{frame} {entry['accuracy']:.1%} ({entry['images']})"
                                                for frame, entry in scores.items())


def none_scores(reads) -> str:
    """From (image, expected, read, confidence): how the cards without power / toughness (NO_VALUE) are read."""
    nones = [read for _, expected, read, _ in reads if expected == data.NO_VALUE]
    others = [read for _, expected, read, _ in reads if expected != data.NO_VALUE]
    right = sum(read == data.NO_VALUE for read in nones)
    wrong = sum(read == data.NO_VALUE for read in others)
    return (f"Without power / toughness read '{data.NO_VALUE}': {right}/{len(nones)}"
            + (f" ({right / len(nones):.1%})" if nones else "")
            + f", with power / toughness read '{data.NO_VALUE}': {wrong}/{len(others)}"
            + (f" ({wrong / len(others):.1%})" if others else ""))


def save_class_scores(rows: List[dict], path: Path):
    columns = ["class", "images", "failed", "failed_percent", "read_as"]
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[column] for column in columns[:-1]]
                            + [" ".join(f"{read}:{count}" for read, count in row["read_as"].items())])


def print_class_scores(rows: List[dict]):
    failed = [row for row in rows if row["failed"]]
    print(f"Classes with failures: {len(failed)}/{len(rows)}" + (", the most failed:" if failed else ""))
    for row in failed[:WORST_CLASSES_PRINTED]:
        read_as = ", ".join(f"{read} x{count}" for read, count in row["read_as"].items())
        print(f"  {row['class']:<9} {row['failed_percent']:5.1f}% failed ({row['failed']}/{row['images']}) read as {read_as}")


def power_toughness_trainer(definition: PowerToughnessTraining, values: Dict[str, str]):
    """
    Ultralytics ClassificationTrainer with:
    - its own dataset: the image (from the RAM cache) augmented by TrainAugmenter (training only), resized to
      input_width x input_height (to_input), with its file ("im_file"), instead of the square crops,
    - a validation that saves the wrong reads (values, not folder names) and the failures of each class, and
      whose fitness is accuracy_top1 (Ultralytics: the mean of top1 and top5).
    values: class folder -> value.
    """
    import functools
    from ultralytics.data import ClassificationDataset, build_dataloader
    from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
    from ultralytics.models.yolo.classify import train as classify_train, val as classify_val

    # No pinned memory: WSL has little, the training stopped on "CUDA error: out of memory" in cuMemHostAlloc
    # (the batches are small, the copy to the GPU is as fast without it).
    classify_train.build_dataloader = classify_val.build_dataloader = functools.partial(build_dataloader,
                                                                                        pin_memory=False)
    width, height = definition.input_width, definition.input_height

    class PowerToughnessDataset(ClassificationDataset):
        def __init__(self, root, args, augment, prefix, names):
            super().__init__(root, args, augment=False, prefix=prefix, names=names)  # Its transforms are not used
            self.augmenter = TrainAugmenter(definition) if augment else None

        def __getitem__(self, i: int) -> dict:
            file, index = self.samples[i][0], self.samples[i][1]
            image = self.img_cache[i] if self.cache_ram else self.imread(file)
            if self.augmenter is not None:
                image = self.augmenter(image)
            return {"img": to_input(image, width, height), "cls": index, "im_file": file}

    class PowerToughnessValidator(ClassificationValidator):
        def build_dataset(self, img_path):
            return PowerToughnessDataset(img_path, self.args, augment=False, prefix=self.args.split, names=self.names)

        def preprocess(self, batch):
            batch = super().preprocess(batch)  # On the device, float
            batch["img"] = batch["img"] / 255
            return batch

        def init_metrics(self, model):
            super().init_metrics(model)
            self.reads = []  # (image, expected, read, confidence) of each image

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            confidences, best = preds.max(1)
            for file, expected, read, confidence in zip(batch["im_file"], batch["cls"].tolist(), best.tolist(),
                                                        confidences.tolist()):
                self.reads.append((file, values[self.names[int(expected)]], values[self.names[int(read)]],
                                   round(float(confidence), 4)))

        def get_stats(self):
            stats = super().get_stats()
            accuracy = stats["metrics/accuracy_top1"]
            rows = class_scores(self.reads)
            errors = [{"image": image, "expected": expected, "read": read, "confidence": confidence}
                      for image, expected, read, confidence in self.reads if read != expected]
            if self.reads and getattr(self, "frames", None) is None:
                self.frames = val_frames(self.reads[0][0])
            frames = frame_scores(self.reads, self.frames or {})
            print(f"Power / toughness read right: {len(self.reads) - len(errors)}/{len(self.reads)} ({accuracy:.1%})")
            print(format_frame_scores(frames))
            print(none_scores(self.reads))
            print_class_scores(rows)
            with open(Path(self.save_dir) / PT_ERRORS_FILE, "w", encoding="utf-8") as fp:
                json.dump({"accuracy": accuracy, "images": len(self.reads), "frames": frames, "classes": rows,
                           "errors": errors}, fp, indent=4, ensure_ascii=False)
            save_class_scores(rows, Path(self.save_dir) / CLASS_ERRORS_FILE)
            stats["fitness"] = accuracy  # Instead of the mean of top1 and top5: chooses best.pt and the early stop
            return stats

    class PowerToughnessTrainer(ClassificationTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            return PowerToughnessDataset(img_path, self.args, augment=mode == "train",
                                         prefix="train" if mode == "train" else self.args.split,
                                         names=self.data["names"])

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)  # On the device
            batch["img"] = batch["img"].float() / 255
            return batch

        def get_validator(self):
            validator = super().get_validator()
            validator.__class__ = PowerToughnessValidator  # Same validator and settings, plus the reads
            return validator

    return PowerToughnessTrainer


def print_summary(yolo_dir: Path, values: Dict[str, str]):
    """The epochs of results.csv (best one marked) and the result of the last validation (best.pt)."""
    results = yolo_dir / "results.csv"
    if results.exists():
        with open(results, "r", encoding="utf-8") as fp:
            rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(fp)]
        best = max(rows, key=lambda row: float(row["metrics/accuracy_top1"]), default=None)
        print(f"{'epoch':>5} {'train/loss':>10} {'val/loss':>9} {'top1':>7} {'top5':>7}")
        for row in rows:
            print(f"{row['epoch']:>5} {float(row['train/loss']):10.4f} {float(row['val/loss']):9.4f} "
                  f"{float(row['metrics/accuracy_top1']):7.1%} {float(row['metrics/accuracy_top5']):7.1%}"
                  + ("  <- best" if row is best else ""))
    errors_path = yolo_dir / PT_ERRORS_FILE
    if errors_path.exists():
        with open(errors_path, "r", encoding="utf-8") as fp:
            scores = json.load(fp)
        print(f"Validation of best.pt: {scores['images']} real cards, read right {scores['accuracy']:.1%}, "
              f"{len(scores['errors'])} errors ({errors_path})")
        if scores.get("frames"):
            print("  " + format_frame_scores(scores["frames"]))
        none = next((row for row in scores["classes"] if row["class"] == data.NO_VALUE), None)
        if none:
            false_none = sum(error["read"] == data.NO_VALUE for error in scores["errors"])
            print(f"  without power / toughness: {none['images'] - none['failed']}/{none['images']} read "
                  f"'{data.NO_VALUE}', {false_none} cards with power / toughness read '{data.NO_VALUE}'")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_classifier(model_path) -> dict:
    """Load a classifier of this file: <hash>/model.pt and the classifier.json next to it."""
    import torch
    from ultralytics import YOLO  # Only needed to train or to classify

    model_path = Path(model_path)
    with open(model_path.parent / CLASSIFIER_INFO_FILE, "r", encoding="utf-8") as fp:
        bundle = json.load(fp)
    bundle["rect"] = training.Rect(*bundle["rect"])
    bundle["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(model_path))
    bundle["names"] = yolo.names  # Class index -> folder
    bundle["model"] = yolo.model.float().eval().to(bundle["device"])
    return bundle


def is_classifier(bundle: dict) -> bool:
    """True for a model of this file (load_classifier), False for a detector (training.load_detector)."""
    return bundle.get("classifier") == "power_toughness"


def window_box(bundle: dict) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the window of the classifier, in CARD_WIDTH x CARD_HEIGHT card coordinates."""
    rect = bundle["rect"]
    x0, y0 = int(rect.center_x - rect.width / 2), int(rect.center_y - rect.height / 2)
    return x0, y0, x0 + int(rect.width), y0 + int(rect.height)


def window_of(bundle: dict, card_bgr: np.ndarray) -> np.ndarray:
    """The window of the classifier on a straightened card (any size, resized to CARD_WIDTH x CARD_HEIGHT)."""
    card_bgr = cv2.resize(card_bgr, (training.CARD_WIDTH, training.CARD_HEIGHT), interpolation=cv2.INTER_LINEAR)
    x0, y0, x1, y1 = window_box(bundle)
    return card_bgr[y0:y1, x0:x1].copy()  # Not a view: the card is freed


def classify_windows(bundle: dict, windows: List[np.ndarray]) -> List[Tuple[str, float]]:
    """(value, confidence) of each window (power_toughness_data_set.IMAGE_WIDTH x IMAGE_HEIGHT BGR images)."""
    import torch

    width, height = bundle["input_size"]
    batch = torch.from_numpy(np.stack([to_input(window, width, height) for window in windows])).to(bundle["device"])
    batch = batch.float() / 255
    with torch.no_grad():
        output = bundle["model"](batch)
    probabilities = output[0] if isinstance(output, (tuple, list)) else output
    confidences, best = probabilities.max(1)
    return [(bundle["values"][bundle["names"][int(index)]], float(confidence))
            for index, confidence in zip(best.tolist(), confidences.tolist())]


def read_power_toughness(bundle: dict, card_bgr: np.ndarray, min_confidence: float = 0.5
                         ) -> Tuple[Optional[str], float]:
    """
    The power / toughness of a straightened card: (value, confidence), value 'none' (power_toughness_data_set.
    NO_VALUE) for a card without power / toughness, (None, confidence) below min_confidence.
    """
    value, confidence = classify_windows(bundle, [window_of(bundle, card_bgr)])[0]
    return (value if confidence >= min_confidence else None), confidence


if __name__ == "__main__":
    for chosen in training._choose_trainings(definitions()):
        chosen.run()
