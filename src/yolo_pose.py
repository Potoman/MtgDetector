"""YOLO Pose (model 5): dataset generation, training and inference.

Uses Ultralytics / PyTorch only. Do not import TensorFlow (or modules that
import it, such as algo_corner_detection, dataset or algo) from this file:
loading TensorFlow and PyTorch in the same process can make their CUDA
libraries clash.
"""
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import tqdm
from ultralytics import YOLO

import download
import local_paths
import noise

# Card corner model, in runs_dir (where it is trained) and in <root>/runs (published for webcam.py).
POSE_MODEL = Path("pose") / "card_corners" / "weights" / "best.pt"
POSE_PUBLISHED_FILES = ["args.yaml", "results.csv", "*.png", "*.jpg", "weights/best.pt"]
# Where the model was trained before local_paths.py (relative project of Ultralytics).
LEGACY_POSE_MODEL = local_paths.ROOT / "runs" / "pose" / "runs" / "pose" / "card_corners" / "weights" / "best.pt"


def _as_list(values) -> Optional[List[str]]:
    """None -> None, "2015" -> ["2015"], ["2003", "2015"] -> ["2003", "2015"]"""
    if values is None:
        return None
    return [values] if isinstance(values, str) else list(values)


def _list_exp_codes(exp_code: Union[str, List[str]]) -> List[str]:
    """'*' = every scryfall/exp_<code>/info.json, otherwise the given code(s)."""
    if exp_code == "*":
        return sorted(p.parent.name[len("exp_"):] for p in download.SCRYFALL_DIR.glob("exp_*/info.json"))
    return _as_list(exp_code)


def _image_path(exp_code: str, illustration_id: str):
    """Where download.download_if_not_exist saves a card image: scryfall/exp_<code>/image/<id>.jpg"""
    return download.SCRYFALL_DIR / f"exp_{exp_code}" / "image" / f"{illustration_id}.jpg"


def select_card_images(count: int, exp_code: Union[str, List[str]] = "*", frame: Union[None, str, List[str]] = None,
                       download_missing: bool = True) -> List[Tuple[str, str, str]]:
    """
    Choose `count` card images in the scryfall folder:
    1. Every expansion of `exp_code` ('*' = all) with at least one card whose 'frame' is in
       `frame` (None = any frame) belongs to the group. N = number of expansions in the group.
    2. Each expansion of the group gives count / N images (the remainder goes to random
       expansions of the group, one more image each), picked at random among its matching cards.
       An expansion with fewer matching cards than its share repeats some of them.
    3. Missing images are downloaded into scryfall/exp_<code>/image/ (download_missing=True).
       With download_missing=False, only the images already in that folder are used.
    :return: the selected images, shuffled: [(exp_code, illustration_id, image_path)]
    """
    frames = _as_list(frame)

    # 1. The group: expansions with at least one matching card image.
    candidates = {}
    for code in _list_exp_codes(exp_code):
        info_path = download.SCRYFALL_DIR / f"exp_{code}" / "info.json"
        if not info_path.exists():
            print(f"Missing {info_path}, expansion '{code}' skipped.")
            continue
        with open(info_path, "r", encoding="utf-8") as fp:
            cards = json.load(fp)
        images = []
        for card in cards:
            if frames is not None and card.get("frame") not in frames:
                continue
            for illustration_id, image_url, _ in download.get_card_images(card):
                path = _image_path(code, illustration_id)
                if image_url and (download_missing or path.exists()):
                    images.append((card["id"], illustration_id))
        if images:
            candidates[code] = images
    if not candidates:
        raise ValueError(f"No card image found for exp_code={exp_code!r} and frame={frame!r}.")

    # 2. count / N images per expansion.
    group = sorted(candidates)
    share, remainder = divmod(count, len(group))
    extra = set(random.sample(group, remainder))
    print(f"{len(group)} expansion(s) with frame {frames if frames else 'any'}: "
          f"{share}{'-' + str(share + 1) if remainder else ''} image(s) each, {count} in total.")

    # 3. Random pick in each expansion, downloading the missing images.
    selected = []
    with tqdm.tqdm(total=count, desc="Select card images", unit="image") as progress:
        for code in group:
            quota = share + (1 if code in extra else 0)
            if quota == 0:
                continue
            pool = candidates[code][:]
            random.shuffle(pool)
            chosen = []
            for card_id, illustration_id in pool:
                if len(chosen) == quota:
                    break
                path = _image_path(code, illustration_id)
                if not path.exists():
                    paths = download.download_if_not_exist(code, card_id)
                    if str(path) not in paths:
                        continue  # Download failed: take another card
                chosen.append((code, illustration_id, str(path)))
                progress.update(1)
            if not chosen:
                print(f"No image available for expansion '{code}', its share is skipped.")
                continue
            # Not enough different cards: repeat some (each one is rendered with a different noise).
            distinct = list(chosen)
            while len(chosen) < quota:
                chosen.append(random.choice(distinct))
                progress.update(1)
            selected.extend(chosen)
    random.shuffle(selected)
    return selected


def _dataset_name(exp_code: Union[str, List[str]], frame: Union[None, str, List[str]]) -> str:
    """Folder name of a dataset, without '*' (not allowed in Windows paths)."""
    exp_part = "all" if exp_code == "*" else "-".join(_as_list(exp_code))
    frames = _as_list(frame)
    frame_part = "frame-" + "-".join(frames) if frames else "frame-any"
    return f"{exp_part}_{frame_part}"


def generate_yolo_pose_dataset(count: int, exp_code: Union[str, List[str]] = "*",
                               frame: Union[None, str, List[str]] = None, val_ratio: float = 0.1,
                               download_missing: bool = True) -> str:
    """
    :param count: number of images in the dataset (n)
    :param exp_code: expansion code, a list of codes, or '*' for every expansion of the scryfall folder
    :param frame: keep only the cards with this 'frame' (ex: "2015" or ["2003", "2015"]). None = any frame
    :param val_ratio: part of the images used for the validation
    :param download_missing: download the selected images missing from the scryfall folder
    """
    output_dir = os.path.join(local_paths.RUNS_DIR, "yolo_pose_dataset", _dataset_name(exp_code, frame), str(count))

    val_count = int(count * val_ratio)
    train_count = count - val_count

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")

    existing_train = len(os.listdir(train_img_dir)) if os.path.isdir(train_img_dir) else 0
    existing_val = len(os.listdir(val_img_dir)) if os.path.isdir(val_img_dir) else 0

    if existing_train >= train_count and existing_val >= val_count:
        print(f"Dataset cache found at {output_dir}, skipping generation.")
        return output_dir

    print(f"Generating dataset at {output_dir}...")
    selected = select_card_images(count, exp_code, frame, download_missing)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    for i, (code, illustration_id, image_path) in enumerate(tqdm.tqdm(selected, desc="Generate dataset", unit="image")):
        split = 'val' if i < val_count else 'train'
        data, bgr = noise.get_noised_mtg_in_background(code, illustration_id, image_path)

        img = cv2.resize(bgr, (400, 400))
        cv2.imwrite(os.path.join(output_dir, "images", split, f"{i}.jpg"), img)

        # Coordinates are in 800×800 space → normalize to [0, 1]
        x0, y0 = data['x0'] / 800, data['y0'] / 800
        x1, y1 = data['x1'] / 800, data['y1'] / 800
        x2, y2 = data['x2'] / 800, data['y2'] / 800
        x3, y3 = data['x3'] / 800, data['y3'] / 800

        xs = [x0, x1, x2, x3]
        ys = [y0, y1, y2, y3]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        bw = x_max - x_min
        bh = y_max - y_min

        # YOLO pose format: class cx cy w h kp1x kp1y kp2x kp2y kp3x kp3y kp4x kp4y
        line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {x0:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f}\n"
        with open(os.path.join(output_dir, "labels", split, f"{i}.txt"), 'w') as f:
            f.write(line)

    return output_dir


def create_yolo_pose_yaml(output_dir: str) -> str:
    abs_path = os.path.abspath(output_dir)
    yaml_path = os.path.join(output_dir, "data.yaml")
    content = (
        f"path: {abs_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['card']\n"
        f"kpt_shape: [4, 2]\n"
    )
    with open(yaml_path, 'w') as f:
        f.write(content)
    return yaml_path


def train_yolo_pose(count: int = 2000, exp_code: Union[str, List[str]] = "*",
                    frame: Union[None, str, List[str]] = None):
    output_dir = generate_yolo_pose_dataset(count, exp_code, frame)
    yaml_path = create_yolo_pose_yaml(output_dir)

    model = YOLO("yolov8s-pose.pt")
    model.train(
        data=yaml_path,
        epochs=400,
        imgsz=400,
        batch=64,
        workers=0,
        project=str(local_paths.RUNS_DIR / "pose"),  # Absolute: the run is in <runs>/pose/card_corners/
        name="card_corners",
        exist_ok=True,
    )
    local_paths.publish(local_paths.RUNS_DIR / "pose" / "card_corners", POSE_PUBLISHED_FILES)
    return model


def load_yolo_pose_model():
    """The card corner model: trained here, else published into <root>/runs, else the one of the previous version."""
    for path in (local_paths.RUNS_DIR / POSE_MODEL, local_paths.PUBLISH_DIR / POSE_MODEL, LEGACY_POSE_MODEL):
        if path.exists():
            return YOLO(str(path))
    raise FileNotFoundError(f"No card corner model: train it first (yolo_pose.train_yolo_pose), {POSE_MODEL}")


def get_yolo_prediction(model, image_bgr: cv2.typing.MatLike) -> dict:
    empty = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'x3': 0, 'y3': 0}
    results = model(image_bgr, verbose=False)[0]
    if results.keypoints is None or len(results.keypoints.xy) == 0:
        return empty
    kps = results.keypoints.xy[0].cpu().numpy()  # shape (4, 2)
    if kps.shape[0] < 4:
        return empty
    return {
        'x0': int(kps[0][0]), 'y0': int(kps[0][1]),
        'x1': int(kps[1][0]), 'y1': int(kps[1][1]),
        'x2': int(kps[2][0]), 'y2': int(kps[2][1]),
        'x3': int(kps[3][0]), 'y3': int(kps[3][1]),
    }


if __name__ == '__main__':
    train_yolo_pose(count=1000, frame="2015")
