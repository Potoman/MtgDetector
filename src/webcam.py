import functools
import json
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Dict, List, Optional, Tuple

import cv2
import numpy as np
import download
import local_paths
import overlay
import power_toughness_data_set
import search
import set_symbol_data_set
import training
import training_corner
import training_mana_cost
import training_power_toughness

# Default MTG card size, in pixels.
CARD_WIDTH = 488
CARD_HEIGHT = 680

# YOLO detectors trained by the training_<...>.py files: runs/detect/<label>/<name>/<hash>/model.pt, and the
# classifiers of training_power_toughness.py: runs/classify/power_toughness/<name>/<hash>/model.pt.
# At launch, one model is chosen for each label (ex: 'set', 'mana_cost', 'power_toughness').
MODEL_ROOT = local_paths.RUNS_DIR / "detect"
CLASSIFIER_ROOT = training_power_toughness.OUTPUT_DIR
PT_LABEL = "power_toughness"
NO_PT = power_toughness_data_set.NO_VALUE   # The power / toughness read on a card without: 'none'
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
WHITE, GRAY, GREEN = (255, 255, 255), (150, 150, 150), (90, 220, 90)
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
    path: Path      # runs/detect/<label>/<name>/<hash>/model.pt, runs/classify/<label>/<name>/<hash>/model.pt

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


def is_classifier_path(path: Path) -> bool:
    """True for a classifier of training_power_toughness.py (its classifier.json next to model.pt)."""
    return (path.parent / training_power_toughness.CLASSIFIER_INFO_FILE).exists()


def list_models() -> Dict[str, List[ModelChoice]]:
    """Every trained model (detectors, power / toughness classifiers), by label, the most recent first."""
    models: Dict[str, List[ModelChoice]] = {}
    for path in MODEL_ROOT.glob("*/*/*/model.pt"):
        label = path.parent.parent.parent.name
        models.setdefault(label, []).append(ModelChoice(label, path))
    for path in CLASSIFIER_ROOT.glob("*/*/model.pt"):
        if is_classifier_path(path):
            models.setdefault(PT_LABEL, []).append(ModelChoice(PT_LABEL, path))
    for choices in models.values():
        choices.sort(key=lambda choice: choice.path.stat().st_mtime, reverse=True)
    return dict(sorted(models.items()))


def load_model(choice: ModelChoice) -> dict:
    """A classifier (training_power_toughness.load_classifier) or a detector (training.load_detector)."""
    if is_classifier_path(choice.path):
        return training_power_toughness.load_classifier(choice.path)
    return training.load_detector(choice.path)


def choose_models() -> List[ModelChoice]:
    """For each label, ask which model to use (Enter = the most recent, 0 = none)."""
    models = list_models()
    if not models:
        print("No trained model found in runs/detect or runs/classify. Run a training first "
              "(ex: 'python3 src/training_set_symbol.py').")
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
    of training_set_symbol.py: 'dsk_rare' -> 'Duskmourn: House of Horror (dsk), rare'), the value otherwise
    ('none' for the power / toughness: 'none (no power / toughness)').
    """
    if label == PT_LABEL and value == NO_PT:
        return "none (no power / toughness)"
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


def read_values(label_models: Dict[str, dict], card_bgr: np.ndarray, drawn_on: np.ndarray
                ) -> Dict[str, Tuple[Optional[str], float]]:
    """
    Every model reads the straightened card card_bgr, in the window of its training (detector.json /
    classifier.json next to model.pt); the window and the boxes found are drawn on drawn_on.
    :return: label -> (value, confidence), value None: nothing read.
    """
    reads = {}
    for label, bundle in label_models.items():
        if training_power_toughness.is_classifier(bundle):
            # The value of the whole window, no box: '2/3', 'none' (no power / toughness).
            value, confidence = training_power_toughness.read_power_toughness(bundle, card_bgr, MIN_CONFIDENCE)
            window, boxes = training_power_toughness.window_box(bundle), []
        elif training_mana_cost.is_mana_cost_detector(bundle):
            # One box per mana symbol, joined into the mana cost: '{1}{W}{W}'.
            value, confidence, boxes = training_mana_cost.read_mana_cost(bundle, card_bgr, MIN_CONFIDENCE)
            window = training.search_window(bundle)
        else:
            value, confidence, box = training.detect(bundle, card_bgr, MIN_CONFIDENCE)
            window, boxes = training.search_window(bundle), [box] if box is not None else []
        draw_detection(drawn_on, window, boxes)
        reads[label] = (value, confidence)
    return reads


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
def find_cards(icon: str, rarity: Optional[str], mana_cost: Optional[str],
               power_toughness: Optional[str] = None) -> List[dict]:
    """
    The cards of the local scryfall data (search.py) printing the set symbol `icon`, with that rarity, mana
    cost and power / toughness ('none': without), None: any, sorted by expansion then collector number.
    Cached: called every frame.
    """
    cards = []
    for exp_code in icon_exp_codes().get(icon, []):
        cards += search.search(search.SearchCriteria(exp_code, rarity, mana_cost, power_toughness))
    return cards


@dataclass
class SearchResult:
    """The search of the card read (run_search): what the panel and the found card show."""
    message: Optional[str] = None           # Why there is no list (no card, a value not read yet...)
    cards: List[dict] = field(default_factory=list)
    power_toughness: Optional[str] = None   # The power / toughness used ('none': without), None: not used
    pt_ignored: bool = False                # The power / toughness read matches no card: searched without it
    found: Optional[dict] = None            # The card found (found_card), None: no card or several


def _oracle_id(card: dict) -> str:
    """The id of the card whatever its printing (expansion, promo...), its name without one."""
    faces = card.get("card_faces") or [{}]
    return card.get("oracle_id") or faces[0].get("oracle_id") or card.get("name", "")


def found_card(cards: List[dict], icon: str) -> Optional[dict]:
    """
    The card found: every card of the list is the same card (same oracle id: an expansion and its promos share
    the set symbol, ex: dsk and pdsk), the printing of the expansion of the symbol if there is one.
    None: no card, or several different cards.
    """
    if not cards or len({_oracle_id(card) for card in cards}) != 1:
        return None
    return next((card for card in cards if card.get("set") == icon), cards[0])


def run_search(read: Dict[str, str], labels: Collection[str], card_detected: bool) -> SearchResult:
    """
    The cards matching what was read, once the set (with its rarity) and the mana cost are read (the power /
    toughness is used once read, not waited for). A power / toughness that matches no card (a wrong read, or an
    old card printed with another value than its JSON one: Clone '★/★', 0/0) is not used.
    :param read: last value read for each label
    :param labels: the labels having a model: the value of a label without model is not waited for
    """
    if not card_detected:
        return SearchResult("No card")
    if "set" not in labels:
        return SearchResult("No 'set' model: no search")
    waiting = [label for label in ("set", "mana_cost") if label in labels and label not in read]
    if waiting:
        return SearchResult(f"Reading the {' and the '.join(label.replace('_', ' ') for label in waiting)}...")
    icon, rarity = set_symbol_data_set.parse_class(read["set"])
    if icon not in icon_exp_codes():
        return SearchResult(f"No expansion of the '{icon}' symbol in scryfall/exp: run src/download.py")
    mana_cost, power_toughness = read.get("mana_cost"), read.get(PT_LABEL)
    cards = find_cards(icon, rarity, mana_cost, power_toughness)
    if power_toughness and not cards:
        cards = find_cards(icon, rarity, mana_cost)
        return SearchResult(cards=cards, pt_ignored=True, found=found_card(cards, icon))
    return SearchResult(cards=cards, power_toughness=power_toughness, found=found_card(cards, icon))


def card_image_path(card: dict, power_toughness: Optional[str]) -> Optional[Path]:
    """
    The image on disk of a card (scryfall/exp/exp_<set>/image/<illustration id>.jpg): of its face whose power /
    toughness is the one read ('none': without), else of its first face. None: the card has no image.
    """
    images = download.get_card_images(card)
    if not images:
        return None
    values = search.image_power_toughness(card)
    wanted = None if power_toughness == NO_PT else power_toughness
    index = next((i for i, value in enumerate(values) if power_toughness and value == wanted and i < len(images)), 0)
    return local_paths.EXP_DIR / f"exp_{card['set']}" / "image" / f"{images[index][0]}.jpg"


@functools.lru_cache(maxsize=32)
def _card_image(path: str) -> Optional[np.ndarray]:
    """A card image of the disk, resized to CARD_WIDTH x CARD_HEIGHT (cached), None if unreadable."""
    image = cv2.imread(path)
    return None if image is None else cv2.resize(image, (CARD_WIDTH, CARD_HEIGHT), interpolation=cv2.INTER_AREA)


def blank_card(lines: List[str]) -> np.ndarray:
    """A black CARD_WIDTH x CARD_HEIGHT image with these lines in the middle (gray)."""
    blank = np.zeros((CARD_HEIGHT, CARD_WIDTH, 3), dtype=np.uint8)
    y = CARD_HEIGHT // 2 - (len(lines) - 1) * HEADER_STEP // 2
    for line in lines:
        text = _ascii(line)
        x = max(PANEL_MARGIN, (CARD_WIDTH - _text_width(text, HEADER_SCALE)) // 2)
        _put_text(blank, text, x, y, HEADER_SCALE, GRAY, CARD_WIDTH - 2 * PANEL_MARGIN)
        y += HEADER_STEP
    return blank


def draw_found_card(result: SearchResult) -> np.ndarray:
    """The right of the triptych: the image of the card found (of the database), else why there is none."""
    if result.found is None:
        if result.message:
            return blank_card([result.message])
        return blank_card(["No card found"] if not result.cards else [f"{len(result.cards)} possible cards",
                                                                     "Not one card yet"])
    path = card_image_path(result.found, result.power_toughness)
    image = _card_image(str(path)) if path is not None else None
    if image is None:
        return blank_card([result.found.get("name", ""), "Image not downloaded (src/download.py)"])
    return image


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


def draw_card_list(result: SearchResult, read: Dict[str, str], labels: Collection[str], height: int) -> np.ndarray:
    """
    The middle of the triptych. On top, what was read: mana cost, power / toughness, rarity and set. Below, the
    cards of the search (run_search), the card found in green, or why there is no list.
    :param read: last value read for each label ('set': '<icon>_<rarity>' or '<icon>', 'mana_cost': '{1}{W}{W}',
                 'power_toughness': '2/3' or 'none')
    :param labels: the labels having a model
    :param height: height of the panel (the one of the reshaped card)
    """
    panel = np.zeros((height, PANEL_WIDTH, 3), dtype=np.uint8)
    right = PANEL_WIDTH - PANEL_MARGIN

    def unknown(label):
        return "?" if label in labels else "(no model)"

    icon, rarity = set_symbol_data_set.parse_class(read["set"]) if "set" in read else (None, None)
    mana_cost, power_toughness = read.get("mana_cost"), read.get(PT_LABEL)
    if icon is None:
        set_text, rarity_text = unknown("set"), unknown("set")
    else:
        set_text = f"{get_expansion_name(icon)} ({icon})"
        rarity_text = rarity or "any (old symbol)"
    if power_toughness is None:
        pt_text = unknown(PT_LABEL)
    else:
        pt_text = "none (no P/T)" if power_toughness == NO_PT else power_toughness
        if result.pt_ignored:
            pt_text += "  (no card has it: not used)"
    header = [("Mana cost", mana_cost or unknown("mana_cost"), WHITE),
              ("P/T", pt_text, GRAY if result.pt_ignored else WHITE),
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

    if result.message:
        return message(result.message)
    cards = result.cards
    if result.found is not None:
        prints = f" ({len(cards)} printings)" if len(cards) > 1 else ""
        _put_text(panel, f"Found: {result.found.get('name', '')}{prints}", PANEL_MARGIN, y, ROW_SCALE, GREEN,
                  right - PANEL_MARGIN)
    else:
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
        bundle = load_model(choice)
        label_models[choice.label] = bundle
        classes = bundle["values"] if training_power_toughness.is_classifier(bundle) else bundle["classes"]
        print(f"'{choice.label}' model loaded: {choice.path} ({len(classes)} classes).")
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
    last_found = None  # Scryfall id of the card found last printed

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
            for label, (value, confidence) in read_values(label_models, reshaped.copy(), reshaped).items():
                if value is not None and value != last_values.get(label):
                    print(f"{label}: {describe_value(label, value)} - {confidence:.0%}")
                    last_values[label] = value

        # Triptych: the card of the webcam, what was read and the possible cards, the card found in the database.
        result = run_search(last_values, label_models.keys(), card_detected)
        found_id = result.found.get("id") if result.found else None
        if found_id and found_id != last_found:
            print(f"Card found: {result.found.get('name')} ({result.found.get('set')} "
                  f"#{result.found.get('collector_number')})")
        last_found = found_id

        cv2.imshow('Original', image_bgr)
        cv2.imshow('Resized', image_bgr_resized)
        cv2.imshow('Red', red)
        panel = draw_card_list(result, last_values, label_models.keys(), reshaped.shape[0])
        cv2.imshow('Reshaped red', np.hstack([reshaped, panel, draw_found_card(result)]))

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
