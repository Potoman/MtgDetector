"""YOLO Pose (model 5): dataset generation, training and inference.

Uses Ultralytics / PyTorch only. Do not import TensorFlow (or modules that
import it, such as algo_corner_detection, dataset or algo) from this file:
loading TensorFlow and PyTorch in the same process can make their CUDA
libraries clash.
"""
import os

import cv2
import tqdm
from ultralytics import YOLO

import card
import noise


def generate_yolo_pose_dataset(exp_code: str, count: int, val_ratio: float = 0.1) -> str:
    output_dir = os.path.join("yolo_pose_dataset", exp_code, str(count))

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
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    for i in tqdm.tqdm(range(count)):
        split = 'val' if i < val_count else 'train'
        data, bgr = noise.get_noised_mtg_in_background(exp_code, card.get_illustration_id_random(exp_code))

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


def train_yolo_pose():
    output_dir = generate_yolo_pose_dataset("mrd", 2000)
    yaml_path = create_yolo_pose_yaml(output_dir)

    model = YOLO("yolov8s-pose.pt")
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=400,
        batch=64,
        workers=0,
        project="runs/pose",
        name="card_corners",
        exist_ok=True,
    )
    return model


def load_yolo_pose_model():
    return YOLO("runs/pose/runs/pose/card_corners/weights/best.pt")


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
    train_yolo_pose()
