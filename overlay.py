import cv2
from typing import Dict


def add_card_border(data: Dict[str, int], img: cv2.typing.MatLike) -> None:
    cv2.line(img, (data['x0'], data['y0']), (data['x1'], data['y1']), color=(0, 0, 255, 255), thickness=3)
    cv2.line(img, (data['x1'], data['y1']), (data['x2'], data['y2']), color=(0, 0, 255, 255), thickness=3)
    cv2.line(img, (data['x2'], data['y2']), (data['x3'], data['y3']), color=(0, 0, 255, 255), thickness=3)
    cv2.line(img, (data['x3'], data['y3']), (data['x0'], data['y0']), color=(0, 0, 255, 255), thickness=3)
