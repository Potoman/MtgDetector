import json
from pathlib import Path

import cv2
import numpy as np
import overlay

# True: YOLO Pose (yolo_pose.py, Ultralytics). False: Keras models 0-4 (algo_corner_detection.py, TensorFlow).
use_yolo = True

# Default MTG card size, in pixels.
CARD_WIDTH = 488
CARD_HEIGHT = 680

ROOT = Path(__file__).resolve().parent.parent

# Expansion classifier trained by training.py: runs/classify/<label>/<name>/<hash>/model.pt
# (TrainingDefinition label="set", name="set"). Each training has its own <hash>/ folder:
# by default the most recent model.pt is used. To use a specific training, set
# SET_MODEL_PATH = SET_MODEL_DIR / "<hash>" / "model.pt".
# Only used with YOLO: training.py uses PyTorch, which must not be loaded with TensorFlow.
SET_MODEL_DIR = ROOT / "runs" / "classify" / "set" / "set"
SET_MODEL_PATH = None  # None = most recent model.pt in SET_MODEL_DIR/<hash>/
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
    """Full expansion name from <root>/scryfall/exp_<exp_code>/info.json (cached). Falls back to the code."""
    if exp_code not in _expansion_names:
        name = exp_code
        try:
            with open(ROOT / "scryfall" / f"exp_{exp_code}" / "info.json", "r", encoding="utf-8") as fp:
                cards = json.load(fp)
            if cards:
                name = cards[0].get("set_name", exp_code)
        except OSError:
            pass
        _expansion_names[exp_code] = name
    return _expansion_names[exp_code]


def find_model(model_path, model_dir):
    """model_path if given, otherwise the most recent model_dir/<hash>/model.pt (None if there is none)."""
    if model_path is not None:
        return model_path if Path(model_path).exists() else None
    models = list(Path(model_dir).glob("*/model.pt"))
    return max(models, key=lambda path: path.stat().st_mtime) if models else None


def draw_rect(card_bgr, rect):
    """Draw the classifier rectangle on the reshaped card, to check its position."""
    x0 = int(round(rect.center_x - rect.width / 2))
    y0 = int(round(rect.center_y - rect.height / 2))
    x1 = int(round(rect.center_x + rect.width / 2))
    y1 = int(round(rect.center_y + rect.height / 2))
    cv2.rectangle(card_bgr, (x0, y0), (x1, y1), (0, 255, 0), 1)


if __name__ == '__main__':
    # Import only the framework we need, so TensorFlow and PyTorch are never loaded together.
    set_classifier = None
    if use_yolo:
        import yolo_pose
        import training
        model = yolo_pose.load_yolo_pose_model()
        set_model_path = find_model(SET_MODEL_PATH, SET_MODEL_DIR)
        if set_model_path is not None:
            set_classifier = training.load_classifier(set_model_path)
            print(f"Expansion model loaded: {set_model_path} ({len(set_classifier['classes'])} expansions).")
        else:
            print(f"No expansion model found in {SET_MODEL_DIR}. Run 'python src/training.py' first.")
    else:
        import algo_corner_detection
        model = algo_corner_detection.load_border_detection_model()
        print("Expansion detection is only available with YOLO (use_yolo = True).")

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Last expansion printed, so the console is not flooded with the same name every frame.
    last_exp_code = None

    while True:
        # Capture frame-by-frame
        ret, image_bgr = cap.read()

        # If frame not read correctly, break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        image_bgr_cropped = image_bgr[:, 80:(80 + 480), :]
        image_bgr_resized = cv2.resize(image_bgr_cropped, (400, 400), interpolation=cv2.INTER_LINEAR)

        if use_yolo:
            prediction = yolo_pose.get_yolo_prediction(model, image_bgr_resized)
        else:
            image_gray = cv2.cvtColor(image_bgr_resized, cv2.COLOR_BGR2GRAY)
            image_norm = image_gray / 255.0
            prediction = algo_corner_detection.get_prediction(model, image_norm)

        red = overlay.add_card_border(prediction, image_bgr_resized)

        # Cut the card from the cropped frame (before resizing to 400x400) for a sharper result.
        scale_x = image_bgr_cropped.shape[1] / image_bgr_resized.shape[1]
        scale_y = image_bgr_cropped.shape[0] / image_bgr_resized.shape[0]
        reshaped = reshape_card(prediction, image_bgr_cropped, scale_x, scale_y)
        if reshaped is None:
            reshaped = no_image()
            last_exp_code = None
        elif set_classifier is not None:
            # Same rectangle as the training: it is saved in model.pt.
            exp_code, confidence = training.predict(set_classifier, reshaped)
            if confidence >= MIN_CONFIDENCE and exp_code != last_exp_code:
                print(f"Expansion: {get_expansion_name(exp_code)} ({exp_code}) - {confidence:.0%}")
                last_exp_code = exp_code
            draw_rect(reshaped, set_classifier["rect"])

        cv2.imshow('Original', image_bgr)
        cv2.imshow('Resized', image_bgr_resized)
        cv2.imshow('Red', red)
        cv2.imshow('Reshaped red', reshaped)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()