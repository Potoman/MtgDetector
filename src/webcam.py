import functools
import json
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Dict, List, Optional

import cv2
import numpy as np
import local_paths
import overlay
import search
import set_symbol_data_set
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

# Panel drawn on the right of the reshaped card (window 'Reshaped red'): the values read and the possible cards.
PANEL_WIDTH = 520
PANEL_MARGIN = 12
PANEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_SCALE, HEADER_STEP = 0.6, 30     # Font scale and line height (pixels) of the values read
VALUE_X = 130                           # x of the values read, after their name
ROW_SCALE, ROW_STEP = 0.5, 21           # Font scale and line height (pixels) of the card list
COLUMN_GAP = 14
WHITE, GRAY = (255, 255, 255), (150, 150, 150)
# Rarity written in the color of the set symbol (BGR).
RARITY_COLORS = {"common": (230, 230, 230), "uncommon": (215, 200, 180), "rare": (60, 190, 230),
                 "mythic": (30, 110, 240)}

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


@functools.lru_cache(maxsize=None)
def icon_exp_codes() -> Dict[str, List[str]]:
    """
    icon -> the downloaded expansions printing that set symbol (scryfall/set/sets.json), ex: 'dsk' ->
    ['adsk', 'dsk', 'pdsk', 'tdsk'], 'tsp' -> ['tsb', 'tsp']: the set detector reads the symbol, not the
    expansion. Without sets.json, the icon is the expansion code.
    """
    try:
        sets = set_symbol_data_set.load_sets()
    except FileNotFoundError:
        sets = {}
    by_icon = defaultdict(list)
    for exp_code in search.list_exp_codes():
        by_icon[set_symbol_data_set.icon_of(sets[exp_code]) if exp_code in sets else exp_code].append(exp_code)
    return dict(by_icon)


@functools.lru_cache(maxsize=64)
def find_cards(icon: str, rarity: Optional[str], mana_cost: Optional[str]) -> List[dict]:
    """
    The cards of the local scryfall data (search.py) printing the set symbol `icon`, with that rarity and mana
    cost (None: any), sorted by expansion then collector number. Cached: called every frame.
    """
    cards = []
    for exp_code in icon_exp_codes().get(icon, []):
        cards += search.search(search.SearchCriteria(exp_code, rarity, mana_cost))
    return cards


def _ascii(text: str) -> str:
    """cv2.putText only draws ASCII: 'Lim-Dûl' -> 'Lim-Dul', 'Æther' -> 'AEther'."""
    text = text.replace("Æ", "AE").replace("æ", "ae")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _text_width(text: str, scale: float) -> int:
    return cv2.getTextSize(text, PANEL_FONT, scale, 1)[0][0]


def _put_text(image, text: str, x: int, y: int, scale: float, color, max_width: int):
    """Write `text` with its baseline at (x, y), cut with '...' to fit in max_width pixels."""
    text = _ascii(text)
    while len(text) > 3 and _text_width(text, scale) > max_width:
        text = text[:-4] + "..."
    cv2.putText(image, text, (x, y), PANEL_FONT, scale, color, 1, cv2.LINE_AA)


def draw_card_list(read: Dict[str, str], labels: Collection[str], card_detected: bool, height: int) -> np.ndarray:
    """
    Panel shown on the right of the reshaped card. On top, what was read: mana cost, rarity and set. Below,
    once the set (with its rarity) and the mana cost are read, the matching cards of the local scryfall data.
    :param read: last value read for each label ('set': '<icon>_<rarity>' or '<icon>', 'mana_cost': '{1}{W}{W}')
    :param labels: the labels having a model: the value of a label without model is not waited for
    :param card_detected: False when no card is detected
    :param height: height of the panel (the one of the reshaped card)
    """
    panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
    right = PANEL_WIDTH - PANEL_MARGIN

    def unknown(label):
        return "?" if label in labels else "(no model)"

    icon, rarity = set_symbol_data_set.parse_class(read["set"]) if "set" in read else (None, None)
    mana_cost = read.get("mana_cost")
    if icon is None:
        set_text, rarity_text = unknown("set"), unknown("set")
    else:
        set_text = f"{get_expansion_name(icon)} ({icon})"
        rarity_text = rarity or "any (old symbol)"
    header = [("Mana cost", mana_cost or unknown("mana_cost"), WHITE),
              ("Rarity", rarity_text, RARITY_COLORS.get(rarity, WHITE)),
              ("Set", set_text, WHITE)]
    y = PANEL_MARGIN
    for name, value, color in header:
        y += HEADER_STEP
        _put_text(panel, name, PANEL_MARGIN, y, HEADER_SCALE, GRAY, VALUE_X - PANEL_MARGIN)
        _put_text(panel, value, VALUE_X, y, HEADER_SCALE, color, right - VALUE_X)
    y += HEADER_STEP // 2
    cv2.line(panel, (PANEL_MARGIN, y), (right, y), GRAY, 1)
    y += HEADER_STEP

    def message(text):
        _put_text(panel, text, PANEL_MARGIN, y, ROW_SCALE, GRAY, right - PANEL_MARGIN)
        return panel

    if not card_detected:
        return message("No card")
    if "set" not in labels:
        return message("No 'set' model: no search")
    waiting = [label for label in ("set", "mana_cost") if label in labels and label not in read]
    if waiting:
        return message(f"Reading the {' and the '.join(label.replace('_', ' ') for label in waiting)}...")
    if icon not in icon_exp_codes():
        return message(f"No expansion of the '{icon}' symbol in scryfall/exp: run src/download.py")

    cards = find_cards(icon, rarity, mana_cost)
    _put_text(panel, f"{len(cards)} possible card(s)", PANEL_MARGIN, y, ROW_SCALE, WHITE, right - PANEL_MARGIN)
    y += ROW_STEP + 4

    # Columns: expansion (only when several), collector number, name.
    max_rows = (height - PANEL_MARGIN - y) // ROW_STEP + 1
    shown = cards if len(cards) <= max_rows else cards[:max_rows - 1]
    several_sets = len({card.get("set") for card in cards}) > 1
    rows = [([card.get("set", "")] if several_sets else []) + [card.get("collector_number", ""), card.get("name", "")]
            for card in shown]
    xs = [PANEL_MARGIN]
    for column in list(zip(*rows))[:-1]:
        xs.append(xs[-1] + max(_text_width(_ascii(value), ROW_SCALE) for value in column) + COLUMN_GAP)
    for row in rows:
        for x, value, color in zip(xs, row, [GRAY] * (len(row) - 1) + [WHITE]):
            _put_text(panel, value, x, y, ROW_SCALE, color, right - x)
        y += ROW_STEP
    if len(shown) < len(cards):
        _put_text(panel, f"... and {len(cards) - len(shown)} more", PANEL_MARGIN, y, ROW_SCALE, GRAY,
                  right - PANEL_MARGIN)
    return panel


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
        card_detected = reshaped is not None
        if not card_detected:
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
        # The card, with on its right what was read and the possible cards.
        panel = draw_card_list(last_values, label_models.keys(), card_detected, reshaped.shape[0])
        cv2.imshow('Reshaped red', np.hstack([reshaped, panel]))

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
