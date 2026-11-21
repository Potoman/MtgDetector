import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import local_paths
import overlay
import training
import training_corner
import training_mana_cost

# Default MTG card size, in pixels.
CARD_WIDTH = 488
CARD_HEIGHT = 680

# YOLO detectors trained by the training_<...>.py files: runs/detect/<label>/<name>/<hash>/model.pt.
# At launch, one model is chosen for each label (ex: 'set', 'mana_cost').
MODEL_ROOT = local_paths.RUNS_DIR / "detect"
# Below this confidence, the prediction is not printed.
MIN_CONFIDENCE = 0.5

_expansion_names = {}


def reshape_card(prediction, image_bgr, scale_x=1.0, scale_y=1.0):
    """
    Warp the quadrilateral found by the model into a flat CARD_WIDTH x CARD_HEIGHT card image.
    :param prediction: corners in the 400x400 image (0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left)
    :param image_bgr: image to cut the card from
    :param scale_x: factor converting prediction x coordinates to image_bgr coordinates
    :param scale_y: factor converting prediction y coordinates to image_bgr coordinates
    :return: the reshaped card, or None if no card was detected
    """
    corners = np.float32([[prediction[f'x{i}'] * scale_x, prediction[f'y{i}'] * scale_y] for i in range(4)])
    # get_yolo_prediction() returns all zeros when no card is found.
    # Also reject degenerate shapes (all corners on one point or one line).
    if cv2.contourArea(corners) < 1.0:
        return None
    target = np.float32([[0, 0], [CARD_WIDTH, 0], [CARD_WIDTH, CARD_HEIGHT], [0, CARD_HEIGHT]])
    m = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(image_bgr, m, (CARD_WIDTH, CARD_HEIGHT))


def no_image():
    """Black CARD_WIDTH x CARD_HEIGHT image with "No image" written in the middle."""
    blank = np.zeros((CARD_HEIGHT, CARD_WIDTH, 3), dtype=np.uint8)
    text = "No image"
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    origin = ((CARD_WIDTH - text_width) // 2, (CARD_HEIGHT + text_height) // 2)
    cv2.putText(blank, text, origin, font, font_scale, (255, 255, 255), thickness)
    return blank


def get_expansion_name(exp_code):
    """Full expansion name from <root>/scryfall/exp/exp_<exp_code>/info.json (cached). Falls back to the code."""
    if exp_code not in _expansion_names:
        name = exp_code
        try:
            with open(local_paths.EXP_DIR / f"exp_{exp_code}" / "info.json", "r", encoding="utf-8") as fp:
                cards = json.load(fp)
            if cards:
                name = cards[0].get("set_name", exp_code)
        except OSError:
            pass
        _expansion_names[exp_code] = name
    return _expansion_names[exp_code]


@dataclass
class ModelChoice:
    label: str      # ex: 'set'
    path: Path      # runs/detect/<label>/<name>/<hash>/model.pt

    def describe(self) -> str:
        """ex: 'set_gen/7fd31c9062  2026-11-01 07:27  100 epochs'"""
        import datetime
        date = datetime.datetime.fromtimestamp(self.path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        text = f"{self.path.parent.parent.name}/{self.path.parent.name}  {date}"
        try:
            with open(self.path.parent / "training.json", "r", encoding="utf-8") as fp:
                text += f"  {json.load(fp)['epochs']} epochs"
        except (OSError, ValueError, KeyError):
            pass
        return text


def list_models() -> Dict[str, List[ModelChoice]]:
    """Every trained model, by label, the most recent first."""
    models: Dict[str, List[ModelChoice]] = {}
    for path in MODEL_ROOT.glob("*/*/*/model.pt"):
        label = path.parent.parent.parent.name
        models.setdefault(label, []).append(ModelChoice(label, path))
    for choices in models.values():
        choices.sort(key=lambda choice: choice.path.stat().st_mtime, reverse=True)
    return dict(sorted(models.items()))


def choose_models() -> List[ModelChoice]:
    """For each label, ask which model to use (Enter = the most recent, 0 = none)."""
    models = list_models()
    if not models:
        print("No trained model found in runs/detect. Run a training first (ex: 'python3 src/training_set_symbol.py').")
    chosen = []
    for label, choices in models.items():
        print(f"Model for '{label}':")
        for index, choice in enumerate(choices, start=1):
            print(f"  {index}. {choice.describe()}")
        print(f"  0. none (no '{label}' detection)")
        while True:
            answer = input("Which one ? (Enter = 1) : ").strip() or "1"
            if answer.isdigit() and 0 <= int(answer) <= len(choices):
                break
            print(f"'{answer}' is not a valid choice.")
        if int(answer) > 0:
            chosen.append(choices[int(answer) - 1])
    return chosen


def describe_value(label, value):
    """
    Text printed for a prediction: the expansion name for 'set' (and the rarity of a set symbol detector
    of training_set_symbol.py: 'dsk_rare' -> 'Duskmourn: House of Horror (dsk), rare'), the value otherwise.
    """
    if label != "set":
        return value
    code, _, rarity = value.rpartition("_")
    if code and rarity in ("common", "uncommon", "rare", "mythic"):
        return f"{get_expansion_name(code)} ({code}), {rarity}"
    return f"{get_expansion_name(value)} ({value})"


def draw_detection(card_bgr, window, boxes):
    """
    Draw the detector search window (gray) and the symbol box(es) it found on the reshaped card,
    alternating green / magenta so two neighbor symbols can be told apart.
    """
    cv2.rectangle(card_bgr, window[:2], (window[2] - 1, window[3] - 1), (160, 160, 160), 1)
    for index, box in enumerate(boxes):
        color = (0, 255, 0) if index % 2 == 0 else (255, 0, 255)
        cv2.rectangle(card_bgr, box[:2], (box[2] - 1, box[3] - 1), color, 1)


if __name__ == '__main__':
    # label -> loaded model
    label_models = {}
    for choice in choose_models():
        bundle = training.load_detector(choice.path)
        label_models[choice.label] = bundle
        print(f"'{choice.label}' model loaded: {choice.path} ({len(bundle['classes'])} classes).")
    # Card corners: YOLO Pose (training_corner.py).
    model = training_corner.load_yolo_pose_model()

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # MJPG, not the default raw YUYV: in WSL the webcam goes through usbipd (USB over the network), too slow
    # for raw frames, which arrive torn (horizontal stripes of different frames). 640x480: the crop below.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Last value printed for each label, so the console is not flooded with the same value every frame.
    last_values = {}

    while True:
        # Capture frame-by-frame
        ret, image_bgr = cap.read()

        # If frame not read correctly, break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        image_bgr_cropped = image_bgr[:, 80:(80 + 480), :]
        image_bgr_resized = cv2.resize(image_bgr_cropped, (400, 400), interpolation=cv2.INTER_LINEAR)

        prediction = training_corner.get_yolo_prediction(model, image_bgr_resized)

        red = overlay.add_card_border(prediction, image_bgr_resized)

        # Cut the card from the cropped frame (before resizing to 400x400) for a sharper result.
        scale_x = image_bgr_cropped.shape[1] / image_bgr_resized.shape[1]
        scale_y = image_bgr_cropped.shape[0] / image_bgr_resized.shape[0]
        reshaped = reshape_card(prediction, image_bgr_cropped, scale_x, scale_y)
        if reshaped is None:
            reshaped = no_image()
            last_values.clear()
        else:
            # Every model reads the same clean card: the rectangles are drawn on `reshaped` only.
            card = reshaped.copy()
            for label, bundle in label_models.items():
                # Same search window as the training: it is saved in detector.json next to model.pt.
                if training_mana_cost.is_mana_cost_detector(bundle):
                    # One box per mana symbol, joined into the mana cost: '{1}{W}{W}'.
                    value, confidence, boxes = training_mana_cost.read_mana_cost(bundle, card, MIN_CONFIDENCE)
                else:
                    value, confidence, box = training.detect(bundle, card, MIN_CONFIDENCE)
                    boxes = [box] if box is not None else []
                draw_detection(reshaped, training.search_window(bundle), boxes)
                if value is not None and value != last_values.get(label):
                    print(f"{label}: {describe_value(label, value)} - {confidence:.0%}")
                    last_values[label] = value

        cv2.imshow('Original', image_bgr)
        cv2.imshow('Resized', image_bgr_resized)
        cv2.imshow('Red', red)
        cv2.imshow('Reshaped red', reshaped)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
