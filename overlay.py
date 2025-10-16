import cv2
from typing import Dict


def add_card_border(data: Dict[str, int], img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    red = img.copy()
    if "is_present" in data and data["is_present"] == 1.0:
        if red.shape == (400, 400, 3):
            cv2.line(red, (data['x0'], data['y0']), (data['x1'], data['y1']), color=(0, 0, 255, 255), thickness=1)
            cv2.line(red, (data['x1'], data['y1']), (data['x2'], data['y2']), color=(0, 0, 255, 255), thickness=1)
            cv2.line(red, (data['x2'], data['y2']), (data['x3'], data['y3']), color=(0, 0, 255, 255), thickness=1)
            cv2.line(red, (data['x3'], data['y3']), (data['x0'], data['y0']), color=(0, 0, 255, 255), thickness=1)
        else:
            cv2.line(red, (data['x0'], data['y0']), (data['x1'], data['y1']), color=(0, 0, 255), thickness=1)
            cv2.line(red, (data['x1'], data['y1']), (data['x2'], data['y2']), color=(0, 0, 255), thickness=1)
            cv2.line(red, (data['x2'], data['y2']), (data['x3'], data['y3']), color=(0, 0, 255), thickness=1)
            cv2.line(red, (data['x3'], data['y3']), (data['x0'], data['y0']), color=(0, 0, 255), thickness=1)
    return red
