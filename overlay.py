import cv2
from typing import Dict


def add_card_border(data: Dict[str, int], img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    red = img.copy()
    cv2.line(red, (data['x0'], data['y0']), (data['x1'], data['y1']), color=(255, 0, 0, 255), thickness=3)
    cv2.line(red, (data['x1'], data['y1']), (data['x2'], data['y2']), color=(255, 0, 0, 255), thickness=3)
    cv2.line(red, (data['x2'], data['y2']), (data['x3'], data['y3']), color=(255, 0, 0, 255), thickness=3)
    cv2.line(red, (data['x3'], data['y3']), (data['x0'], data['y0']), color=(255, 0, 0, 255), thickness=3)
    return red
