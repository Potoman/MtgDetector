import card
import cv2
import noise
import numpy as np
import tensorflow as tf


def generate_data_set(count) -> tf.data.Dataset:
    imgs_train = np.zeros((count, 800, 800, 3), dtype=np.float32)
    labels_train = np.zeros((count, 9), dtype=np.int32)

    exp_code: str = "mrd"

    index = 0
    for index in range(count):
        data, background_rgba = noise.get_noised_mtg_in_background(exp_code, card.get_illustration_id_random(exp_code))
        background_bgr = cv2.cvtColor(background_rgba, cv2.COLOR_RGBA2BGR)
        imgs_train[index] = background_bgr / 255.0
        labels_train[index] = [1.0, data['x0'], data['x1'], data['x2'], data['x3'], data['y0'], data['y0'], data['y0'], data['y0']]
        index = index + 1

    return tf.data.Dataset.from_tensor_slices((imgs_train, labels_train))


if __name__ == '__main__':
    generate_data_set(10)
    print("pass")

