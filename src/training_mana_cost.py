"""
Mana cost detection: a YOLO detector that finds each mana symbol of the mana cost, at the top right
of a straightened card. One box and one class per {...} symbol of the mana cost: '{1}{W}{W}' -> 3 boxes,
classes '1', 'W', 'W'. Uses the YOLO detect steps of training.py (stamps, training, publishing).

'mana_cost_gen' (ManaCostGenTraining): trained on generated images only and validated on real cards
only, with the dataset of mana_cost_data_set.py (runs/detect/mana_cost/gen/<n>/, n asked at launch,
built only if its dataset.stamp doesn't match). Search window: 300 x 110 at the top right.
Its validation also reads the mana cost of each real card (mana_cost_trainer): metrics/mana_cost_acc, the
share of cards read exactly right, chooses best.pt. mAP and losses are still computed on the boxes.

Answer of the model: the symbols found, joined from left to right: '{1}{W}{W}' (read_mana_cost).
Files: runs/detect/mana_cost/<name>/<hash>/, as every TrainingYoloDetect (see training.py).
Run from anywhere: python3 src/training_mana_cost.py (menu of these trainings)
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import training
from training import Box, Rect, TrainingYoloDetect

# ---------------------------------------------------------------------------
# Cards (shared with mana_cost_data_set.py)
# ---------------------------------------------------------------------------

FRAMES = ["2003", "2015"]
LAYOUTS = ["normal", "flip", "transform", "modal_dfc", "leveler"]
ACCEPT = {"frame": FRAMES, "layout": LAYOUTS}
# Planeswalkers: other frame. Playtest cards (unk, cmb1, cmb2, mb2...): placeholder art or mana cost elsewhere.
FILTERS = {"type_line": ["Legendary Planeswalker", "Planeswalker"], "promo_types": ["playtest"], "set": ["otp"]}

# ---------------------------------------------------------------------------
# Symbols (shared with mana_cost_data_set.py)
# ---------------------------------------------------------------------------


SYMBOL_PATTERN = re.compile(r"\{[^}]+\}")


def symbols_of(value: str) -> List[str]:
    """Classes of a mana cost, from left to right: '{2}{W/U}' -> ['2', 'W/U']."""
    return [symbol[1:-1] for symbol in SYMBOL_PATTERN.findall(value)]


def join_symbols(classes: List[str]) -> str:
    """Inverse of symbols_of: ['1', 'W', 'W'] -> '{1}{W}{W}'."""
    return "".join(f"{{{name}}}" for name in classes)


# ---------------------------------------------------------------------------
# Training definition
# ---------------------------------------------------------------------------


@dataclass
class ManaCostGenTraining(TrainingYoloDetect):
    """
    The mana cost detector trained on generated images only and validated on real cards only: the
    dataset of mana_cost_data_set.py, runs/detect/mana_cost/gen/<samples>/, built unless its
    dataset.stamp matches (shared by every training of the same samples). rect must be its VAL_RECT
    (the window of the validation cards, which becomes the search window of the detector), jitter 0.
    blur_kernels and downscale are not used (the dataset has no copies): use `augmentations`.
    """
    samples: int = 0    # n of mana_cost_data_set.py. 0 = asked at launch

    KIND = "YOLO detect, mana cost, generated images"

    def extra_fields(self) -> dict:
        import mana_cost_data_set as gen  # Not at the top of the file: it imports this file
        return {"type": "mana_cost_gen", "samples": self.samples, "dataset_version": gen.VERSION}

    def run(self) -> Optional[Path]:
        return process_mana_cost_gen_training(self)


def definitions() -> List[ManaCostGenTraining]:
    """The mana cost trainings, in the order of the menu."""
    return [
        # Trained on the generated images of mana_cost_data_set.py, validated on real cards. n is
        # asked at launch. rect = mana_cost_data_set.VAL_RECT: 300 x 110 at the top right of the card.
        ManaCostGenTraining(
            name="mana_cost_gen",
            label="mana_cost",
            sub_categories=[],
            rect=Rect(center_x=338, center_y=55, height=110, width=300),
            accept=ACCEPT,      # The validation cards (see mana_cost_data_set.py)
            filters=FILTERS,
            input_size=320,
            jitter=0.0,
            epochs=15,
            patience=5,
            batch_size=32,
            cache="ram",
            translate=0.0,
            mosaic=0.0,
            # Lighter color changes than the Ultralytics defaults (0.015, 0.7, 0.4): the hue never moves
            # and the Phyrexian symbols ({U/P}, {B/P}...) differ only by their color.
            hsv_h=0.0,
            hsv_s=0.3,
            hsv_v=0.2,
            # The webcam: a blur on a third of the images, white balance (light color temperature),
            # brightness, contrast and gamma. ToGray is off: the Phyrexian symbols differ only by their color.
            augmentations=[
                ("Blur", {"blur_limit": (3, 5), "p": 0.33}),
                ("PlanckianJitter", {"temperature_limit": (3500, 9000), "p": 0.5}),
                ("RandomBrightnessContrast", {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5}),
                ("RandomGamma", {"gamma_limit": (80, 120), "p": 0.3}),
                ("ToGray", {"p": 0.0}),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def process_mana_cost_gen_training(definition: ManaCostGenTraining) -> Optional[Path]:
    """Ask n if needed, load or build the dataset of mana_cost_data_set.py, then train on it."""
    import dataclasses
    import mana_cost_data_set as gen  # Not at the top of the file: it imports this file

    if tuple(definition.rect) != tuple(gen.VAL_RECT) or definition.jitter:
        raise ValueError(f"[{definition.name}] rect must be mana_cost_data_set.VAL_RECT {tuple(gen.VAL_RECT)} "
                         f"and jitter 0: the detector searches where the validation cards were cut")
    if definition.samples <= 0:
        definition = dataclasses.replace(definition, samples=gen.ask_samples())
    training.start_training(definition)
    yaml_path, classes = gen.load_or_build_dataset(definition.samples)
    return training.train_yolo_detect(definition, yaml_path, classes,
                                      extra_info={"detector": "mana_cost", "dataset": str(yaml_path.parent)},
                                      trainer=mana_cost_trainer())


# ---------------------------------------------------------------------------
# Validation on the mana cost read
# ---------------------------------------------------------------------------

READ_CONFIDENCE = 0.5   # Lowest confidence of a symbol read: MIN_CONFIDENCE of webcam.py
READ_IOU = 0.7          # Two boxes overlapping more than this are one symbol: the most confident is kept
MANA_COST_ERRORS_FILE = "mana_cost_errors.json"  # In the Ultralytics run folder (<hash>/yolo/)


def _read_boxes(names: dict, bboxes, classes, confidences=None) -> str:
    """
    Mana cost of the boxes of one image, joined from left to right ('{1}{W}{W}', '' = no symbol).
    With confidences (predictions), as the webcam reads it: only the boxes of at least READ_CONFIDENCE,
    one per place (the most confident one, whatever its class). Without (labels): every box.
    """
    if confidences is not None and len(confidences):
        import torchvision  # Installed with ultralytics

        keep = confidences >= READ_CONFIDENCE
        bboxes, classes, confidences = bboxes[keep], classes[keep], confidences[keep]
        keep = torchvision.ops.nms(bboxes.float(), confidences.float(), READ_IOU)
        bboxes, classes = bboxes[keep], classes[keep]
    elif confidences is not None:
        return ""
    found = sorted(zip(bboxes.tolist(), classes.tolist()), key=lambda item: item[0][0] + item[0][2])
    return join_symbols([names[int(name)] for _, name in found])


def mana_cost_trainer():
    """
    Ultralytics DetectionTrainer whose validation also reads the mana cost of each image and compares
    it with the one of its labels (only their classes and their order count, not where the boxes are):
    metrics/mana_cost_acc = images read exactly right (printed, in results.csv). It is the fitness, so
    it chooses best.pt and stops the training (patience). mAP and the losses are still computed on the
    boxes, as before. The wrong reads of the last validation (best.pt at the end) are saved in
    <hash>/yolo/mana_cost_errors.json.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

    class ManaCostValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.mana_cost_reads = []  # (image, expected, read) of each image

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                expected = _read_boxes(self.names, pbatch["bboxes"], pbatch["cls"])
                read = _read_boxes(self.names, pred["bboxes"], pred["cls"], pred["conf"])
                self.mana_cost_reads.append((pbatch["im_file"], expected, read))

        def get_stats(self):
            stats = super().get_stats()
            total = len(self.mana_cost_reads)
            errors = [{"image": image, "expected": expected, "read": read}
                      for image, expected, read in self.mana_cost_reads if read != expected]
            accuracy = (total - len(errors)) / total if total else 0.0
            print(f"Mana cost read exactly right: {total - len(errors)}/{total} ({accuracy:.1%})")
            with open(Path(self.save_dir) / MANA_COST_ERRORS_FILE, "w", encoding="utf-8") as fp:
                json.dump({"accuracy": accuracy, "images": total, "errors": errors}, fp, indent=4, ensure_ascii=False)
            stats["metrics/mana_cost_acc"] = accuracy
            stats["fitness"] = accuracy  # Instead of the mAP: chooses best.pt and the early stop
            return stats

    class ManaCostTrainer(DetectionTrainer):
        def get_validator(self):
            validator = super().get_validator()
            validator.__class__ = ManaCostValidator  # Same validator and settings, plus the mana cost read
            return validator

    return ManaCostTrainer


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
