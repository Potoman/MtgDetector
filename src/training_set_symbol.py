"""
Set symbol detection: a YOLO detector that finds the set symbol (expansion symbol) on the right of the
type line of a straightened card, its box, and the rarity shown by its color. One class per set and
rarity, '<icon>_<rarity>' (ex: 'dsk_rare'), only '<icon>' before Exodus (ex: 'arn'): see
set_symbol_data_set.class_name. Uses the YOLO detect steps of training.py.

'set_gen' (SetSymbolGenTraining): trained on generated images only and validated on real cards only,
with the dataset of set_symbol_data_set.py (runs/detect/set/gen/<n>_val<percent>_nosym<percent>/: n, the
percent of validation and the percent without symbol asked at launch, built only if its dataset.stamp
doesn't match). The generated symbols are the PNG of colorize_set_symbol.py (the symbol 27 px high), pasted
as they are: the brightness, saturation, gamma, blur... changes come from the training (augmentations, hsv), as
for the mana cost. Search window: 150 x 150 at the right end of the type line (every frame, 1993 to 2015).
The validation cards have no box (the place of a real symbol is not known): the validation reads the
class of each card (set_symbol_trainer, with the "class" of images.json) and only that counts:
metrics/symbol_acc, the share of cards read right, set and rarity (the most confident box, nothing on a
card without symbol), chooses best.pt and stops the training. Also printed: metrics/set_acc (the set
only) and metrics/rarity_acc (the rarity only, sets since Exodus), and the most failed sets (see
set_scores: the failure percent of each set, saved in <hash>/yolo/set_errors.csv and symbol_errors.json;
at the end of the training <hash>/yolo/set_failures.txt: the share of the cards of each set where the set
was not read, whatever the rarity).
The mAP of the validation stays 0 (Ultralytics warns "no labels found"): normal. The box losses of the
training are computed as usual.
Files: runs/detect/set/set_gen/<hash>/, as every TrainingYoloDetect (see training.py). webcam.py lists
the model with the other 'set' models.
Run from anywhere: python3 src/training_set_symbol.py (menu of these trainings)
"""
import csv
import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import training
import set_symbol_data_set as data
from training import TrainingYoloDetect


@dataclasses.dataclass
class SetSymbolGenTraining(TrainingYoloDetect):
    """
    The set symbol detector trained on generated images only and validated on real cards only: the
    dataset of set_symbol_data_set.py, runs/detect/set/gen/<samples>_val<val_percent>_nosym<no_symbol_percent>/,
    built unless its dataset.stamp matches. rect must be its VAL_RECT (the window of the validation cards,
    which becomes the search window of the detector), jitter 0. blur_kernels and downscale are not used:
    use `augmentations`.
    """
    samples: int = 0                    # n of set_symbol_data_set.py. 0 = asked at launch
    val_percent: float = 0.0            # Percent of the samples for the validation (real cards). 0 = asked at launch
    no_symbol_percent: float = -1.0     # Percent of each split without symbol (old cards). -1 = asked at launch

    KIND = "YOLO detect, set symbol, generated images"

    def extra_fields(self) -> dict:
        return {"type": "set_symbol_gen", "samples": self.samples, "val_percent": self.val_percent,
                "no_symbol_percent": self.no_symbol_percent, "dataset_version": data.VERSION,
                "symbols": data.drawing_stamps()}

    def run(self) -> Optional[Path]:
        return process_set_symbol_gen_training(self)


def definitions() -> List[SetSymbolGenTraining]:
    """The set symbol trainings, in the order of the menu."""
    return [
        # Trained on the generated images of set_symbol_data_set.py, validated on real cards. n and the
        # percents are asked at launch. rect = set_symbol_data_set.VAL_RECT: 150 x 150 at the right of the type line.
        SetSymbolGenTraining(
            name="set_gen",
            label="set",
            sub_categories=[],
            rect=data.VAL_RECT,
            accept={"frame": data.FRAMES},  # The validation cards (see set_symbol_data_set.py)
            filters={},
            model="yolo26s.pt",  # Small instead of the nano default (training.TrainingYoloDetect.model)
            input_size=160,  # YOLO imgsz: a multiple of 32, the 150 x 150 window is resized to it
            jitter=0.0,
            epochs=80,
            patience=40,
            batch_size=64,
            cache="ram",
            translate=0.0,
            mosaic=0.0,
            # As mana_cost_gen (training_mana_cost.py): lighter color changes than the Ultralytics defaults,
            # a blur on a third of the images (the webcam is blurry), gamma. No white balance (PlanckianJitter:
            # a warm light turns the silver of uncommon into the gold of rare) and no RandomBrightnessContrast
            # (its contrast is the gain of hsv_v, its brightness turns the black of common into grey).
            hsv_h=0.0,
            hsv_s=0.4,
            hsv_v=0.3,
            augmentations=[
                ("Blur", {"blur_limit": (3), "p": 0.5}),
                ("RandomGamma", {"gamma_limit": (80, 120), "p": 0.3}),
                ("ToGray", {"p": 0.0}),
            ],
        ),
    ]


def val_augmentation(definition: SetSymbolGenTraining) -> dict:
    """
    The augmentation of the real cards of the validation: the color and blur changes of the training
    (hsv_h, hsv_s, hsv_v, augmentations), made once when the dataset is built (see set_symbol_data_set.py).
    """
    return data.val_augmentation_of(definition.hsv_h, definition.hsv_s, definition.hsv_v,
                                    definition.augmentations)


def process_set_symbol_gen_training(definition: SetSymbolGenTraining) -> Optional[Path]:
    """Ask n and the percents if needed, load or build the dataset of set_symbol_data_set.py, then train."""
    if tuple(definition.rect) != tuple(data.VAL_RECT) or definition.jitter:
        raise ValueError(f"[{definition.name}] rect must be set_symbol_data_set.VAL_RECT {tuple(data.VAL_RECT)} "
                         f"and jitter 0: the detector searches where the validation cards were cut")
    if definition.samples <= 0:
        definition = dataclasses.replace(definition, samples=data.ask_samples())
    if definition.val_percent <= 0:
        definition = dataclasses.replace(definition, val_percent=data.ask_val_percent(definition.samples))
    if definition.no_symbol_percent < 0:
        definition = dataclasses.replace(definition, no_symbol_percent=data.ask_no_symbol_percent(
            definition.samples, definition.val_percent))
    output_dir = training.start_training(definition)
    yaml_path, classes = data.load_or_build_dataset(definition.samples, definition.val_percent,
                                                    definition.no_symbol_percent,
                                                    val_augmentation=val_augmentation(definition))
    model_path = training.train_yolo_detect(definition, yaml_path, classes,
                                            extra_info={"detector": "set_symbol", "dataset": str(yaml_path.parent)},
                                            trainer=set_symbol_trainer())
    save_set_failures(output_dir / "yolo")
    return model_path


# ---------------------------------------------------------------------------
# Validation on the class read
# ---------------------------------------------------------------------------

READ_CONFIDENCE = 0.5   # Lowest confidence of a symbol read: MIN_CONFIDENCE of webcam.py
SYMBOL_ERRORS_FILE = "symbol_errors.json"  # In the Ultralytics run folder (<hash>/yolo/)
SET_ERRORS_FILE = "set_errors.csv"         # Same folder: the failure percent of each set (see set_scores)
SET_FAILURES_FILE = "set_failures.txt"     # Same folder, at the end of the training: the set failures (save_set_failures)
IMAGES_FILE = "images.json"                # Of the dataset (set_symbol_data_set.py): the class of each image
NO_SYMBOL_SET = "(no symbol)"              # The "set" of the cards without symbol in set_scores
WORST_SETS_PRINTED = 10                    # Sets printed after each validation, the most failed first
READ_AS_KEPT = 5                           # Wrong reads kept for each set, the most frequent first


def expected_classes(image_file: str) -> Dict[str, str]:
    """
    The class of each image of the dataset of image_file (its images.json, in the folder above images/),
    by absolute path of the image: '<icon>_<rarity>', '<icon>', '' = no symbol.
    """
    root = Path(image_file).resolve().parents[2]  # <dataset>/images/<split>/<name>.png
    with open(root / IMAGES_FILE, "r", encoding="utf-8") as fp:
        entries = json.load(fp)
    return {str((root / entry["file"]).resolve()): entry["class"] or "" for entry in entries}


def read_class(names: dict, classes, confidences) -> str:
    """The class read on one image: the one of the most confident box, if at least READ_CONFIDENCE (as
    webcam.py reads it), '' = no symbol."""
    if not len(confidences):
        return ""
    best = int(confidences.argmax())
    return names[int(classes[best])] if float(confidences[best]) >= READ_CONFIDENCE else ""


def read_scores(reads) -> dict:
    """
    From (image, expected, read) classes: 'symbol' (set and rarity right, '' read on a card without symbol),
    'set' (set right), 'rarity' (rarity right, among the cards of a set since Exodus that have a symbol and
    where one was read).
    """
    total = len(reads)
    symbol = sum(read == expected for _, expected, read in reads)
    sets = sum(data.parse_class(read)[0] == data.parse_class(expected)[0] if expected else read == ""
               for _, expected, read in reads)
    with_rarity = [(data.parse_class(expected)[1], data.parse_class(read)[1]) for _, expected, read in reads
                   if expected and read and data.parse_class(expected)[1]]
    rarity = sum(expected == read for expected, read in with_rarity)
    return {"symbol": symbol / total if total else 0.0, "set": sets / total if total else 0.0,
            "rarity": rarity / len(with_rarity) if with_rarity else 0.0, "images": total}


def set_scores(reads) -> List[dict]:
    """
    From (image, expected, read) classes: the failures of each set (the icon of the expected class,
    NO_SYMBOL_SET for the cards without symbol), the most failed first (percent, then images).
    'failed' = read != expected, split in 'missed' (nothing read), 'wrong_set' (another set read) and
    'wrong_rarity' (the right set, another rarity); on a card without symbol every failure is a wrong_set.
    'read_as': the READ_AS_KEPT most frequent wrong reads ('' = nothing).
    """
    by_set: Dict[str, dict] = {}
    for _, expected, read in reads:
        icon = data.parse_class(expected)[0] if expected else NO_SYMBOL_SET
        entry = by_set.setdefault(icon, {"set": icon, "images": 0, "failed": 0, "failed_percent": 0.0, "missed": 0,
                                         "wrong_set": 0, "wrong_rarity": 0, "read_as": Counter()})
        entry["images"] += 1
        if read == expected:
            continue
        entry["failed"] += 1
        entry["read_as"][read] += 1
        if not read:
            entry["missed"] += 1
        elif expected and data.parse_class(read)[0] == icon:
            entry["wrong_rarity"] += 1
        else:
            entry["wrong_set"] += 1
    rows = []
    for entry in by_set.values():
        entry["failed_percent"] = round(100 * entry["failed"] / entry["images"], 1)
        entry["read_as"] = dict(entry["read_as"].most_common(READ_AS_KEPT))
        rows.append(entry)
    return sorted(rows, key=lambda row: (-row["failed_percent"], -row["images"], row["set"]))


def save_set_scores(rows: List[dict], path: Path):
    """The rows of set_scores in a CSV (a spreadsheet opens it), read_as as 'class:count' separated by ' '."""
    columns = ["set", "images", "failed", "failed_percent", "missed", "wrong_set", "wrong_rarity", "read_as"]
    with open(path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(columns)
        for row in rows:
            read_as = " ".join(f"{read or 'nothing'}:{count}" for read, count in row["read_as"].items())
            writer.writerow([row[column] for column in columns[:-1]] + [read_as])


def print_set_scores(rows: List[dict]):
    """The WORST_SETS_PRINTED most failed sets, with the number of sets with at least one failure."""
    failed = [row for row in rows if row["failed"]]
    print(f"Sets with failures: {len(failed)}/{len(rows)}" + (", the most failed:" if failed else ""))
    for row in failed[:WORST_SETS_PRINTED]:
        read_as = ", ".join(f"{read or 'nothing'} x{count}" for read, count in row["read_as"].items())
        print(f"  {row['set']:<12} {row['failed_percent']:5.1f}% failed ({row['failed']}/{row['images']}: "
              f"missed {row['missed']}, wrong set {row['wrong_set']}, wrong rarity {row['wrong_rarity']}) "
              f"read as {read_as}")


def set_names() -> Dict[str, str]:
    """The name of each set icon (the set whose code is the icon, else the first one), {} without sets.json."""
    try:
        sets = data.load_sets()
    except FileNotFoundError:
        return {}
    names: Dict[str, str] = {}
    for code, set_data in sets.items():
        icon = data.icon_of(set_data)
        if code == icon or icon not in names:
            names[icon] = set_data.get("name") or ""
    return names


def write_set_failures(rows: List[dict], path: Path, header: str = ""):
    """
    The rows of set_scores in a text file: for each set, the share of its cards where the set was not read
    right, whatever the rarity ('missed' + 'wrong_set': nothing read, or another set), the most failed first,
    with the sets read instead. header: the first lines.
    """
    lines = []
    for row in rows:
        failed = row["missed"] + row["wrong_set"]
        read_as: Counter = Counter()
        for read, count in row["read_as"].items():  # The wrong reads of another set (not of a rarity)
            icon = data.parse_class(read)[0] if read else "nothing"
            if icon != row["set"]:
                read_as[icon] += count
        lines.append((-failed / row["images"], -row["images"], row["set"], failed, row, read_as))
    lines.sort(key=lambda line: line[:3])
    names = set_names()
    failing = sum(1 for line in lines if line[3])
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(header + f"Sets not read right (nothing read or another set, whatever the rarity): "
                          f"{failing}/{len(lines)} sets\n\n")
        fp.write(f"{'set':<12} {'failed':>7} {'cards':>7}  {'missed':>6} {'other':>6}  {'name':<36} read as\n")
        for _, _, icon, failed, row, read_as in lines:
            cards = f"{failed}/{row['images']}"
            reads = ", ".join(f"{read} x{count}" for read, count in read_as.most_common())
            fp.write(f"{icon:<12} {100 * failed / row['images']:6.1f}% {cards:>7}  {row['missed']:>6} "
                     f"{row['wrong_set']:>6}  {names.get(icon, '')[:36]:<36} {reads}".rstrip() + "\n")


def save_set_failures(yolo_dir: Path) -> Optional[Path]:
    """
    At the end of the training: SET_FAILURES_FILE in yolo_dir (<hash>/yolo/), the set failures of its
    last validation (best.pt), from the "sets" of its SYMBOL_ERRORS_FILE (write_set_failures).
    :return: the file, None without SYMBOL_ERRORS_FILE.
    """
    errors_path = yolo_dir / SYMBOL_ERRORS_FILE
    if not errors_path.exists():
        print(f"No {errors_path}: no {SET_FAILURES_FILE}.")
        return None
    with open(errors_path, "r", encoding="utf-8") as fp:
        scores = json.load(fp)
    path = yolo_dir / SET_FAILURES_FILE
    write_set_failures(scores["sets"], path,
                       header=f"Validation of best.pt: {scores['images']} cards, set read right {scores['set']:.1%}, "
                              f"set + rarity {scores['symbol']:.1%}\n")
    print(f"Set failures: {path}")
    return path


def set_symbol_trainer():
    """
    Ultralytics DetectionTrainer whose validation reads the class of each image and compares it with the
    "class" of images.json (the validation images have no box): metrics/symbol_acc = images read right,
    set and rarity (printed, in results.csv), with metrics/set_acc and metrics/rarity_acc (see read_scores).
    symbol_acc is the fitness, so it chooses best.pt and stops the training (patience). The wrong reads of
    the last validation are saved in <hash>/yolo/symbol_errors.json, with the failures of each set
    (set_scores, also in <hash>/yolo/set_errors.csv). The last validation is the one of best.pt, after
    the training.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

    class SetSymbolValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.symbol_reads = []  # (image, expected, read) of each image
            self.expected = None    # File -> class, read from images.json with the first image

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                image_file = batch["im_file"][si]
                if self.expected is None:
                    self.expected = expected_classes(image_file)
                expected = self.expected.get(str(Path(image_file).resolve()))
                if expected is None:
                    raise KeyError(f"{image_file} is not in the {IMAGES_FILE} of its dataset: build it again.")
                read = read_class(self.names, pred["cls"], pred["conf"])
                self.symbol_reads.append((image_file, expected, read))

        def get_stats(self):
            stats = super().get_stats()
            scores = read_scores(self.symbol_reads)
            errors = [{"image": image, "expected": expected, "read": read}
                      for image, expected, read in self.symbol_reads if read != expected]
            per_set = set_scores(self.symbol_reads)
            print(f"Read right: set + rarity {scores['symbol']:.1%}, set {scores['set']:.1%}, "
                  f"rarity {scores['rarity']:.1%} ({scores['images']} images)")
            print_set_scores(per_set)
            with open(Path(self.save_dir) / SYMBOL_ERRORS_FILE, "w", encoding="utf-8") as fp:
                json.dump({**scores, "sets": per_set, "errors": errors}, fp, indent=4, ensure_ascii=False)
            save_set_scores(per_set, Path(self.save_dir) / SET_ERRORS_FILE)
            stats["metrics/symbol_acc"] = scores["symbol"]
            stats["metrics/set_acc"] = scores["set"]
            stats["metrics/rarity_acc"] = scores["rarity"]
            stats["fitness"] = scores["symbol"]  # Instead of the mAP: chooses best.pt and the early stop
            return stats

    class SetSymbolTrainer(DetectionTrainer):
        def get_validator(self):
            validator = super().get_validator()
            validator.__class__ = SetSymbolValidator  # Same validator and settings, plus the class read
            return validator

    return SetSymbolTrainer


if __name__ == "__main__":
    for chosen in training._choose_trainings(definitions()):
        chosen.run()
