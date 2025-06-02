import card
import cv2
import noise
import numpy as np
import tensorflow as tf
import tqdm


def generate_data_set(count) -> tf.data.Dataset:
    imgs_train = np.zeros((count, 400, 400, 3), dtype=np.float32)
    labels_train = np.zeros((count, 8), dtype=np.float32)

    exp_code: str = "mrd"

    index = 0
    for index in tqdm.tqdm(range(count)):
        img_train, label_train = get_image_and_keypoint(exp_code)
        imgs_train[index] = img_train
        labels_train[index] = label_train
        index = index + 1

    return tf.data.Dataset.from_tensor_slices((imgs_train, labels_train))


def generate_gray_dataset(exp_code: str, count: int) -> tf.data.Dataset:
    imgs_train = np.zeros((count, 400, 400, 1), dtype=np.float32)
    labels_train = np.zeros((count, 8), dtype=np.float32)

    index = 0
    for index in tqdm.tqdm(range(count)):
        img_train, label_train = get_norm_image_and_keypoint(exp_code)
        imgs_train[index] = img_train.reshape((400, 400, 1))
        labels_train[index] = label_train
        index = index + 1

    return tf.data.Dataset.from_tensor_slices((imgs_train, labels_train))


def get_image_and_keypoint(exp_code: str):
    data, background_rgba = noise.get_noised_mtg_in_background(exp_code, card.get_illustration_id_random(exp_code))
    background_bgr = cv2.cvtColor(background_rgba, cv2.COLOR_RGBA2BGR)
    background_bgr = cv2.resize(background_bgr, (400, 400), interpolation=cv2.INTER_LINEAR)
    img_train = background_bgr / 255.0
    label_train = [data['x0'] / 800.0,
                   data['x1'] / 800.0,
                   data['x2'] / 800.0,
                   data['x3'] / 800.0,
                   data['y0'] / 800.0,
                   data['y1'] / 800.0,
                   data['y2'] / 800.0,
                   data['y3'] / 800.0]
    return img_train, label_train


def get_norm_image_and_keypoint(exp_code: str):
    data, background_rgba = noise.get_noised_mtg_in_background(exp_code, card.get_illustration_id_random(exp_code))
    background_bgr = cv2.cvtColor(background_rgba, cv2.COLOR_BGRA2GRAY)
    background_bgr = cv2.resize(background_bgr, (400, 400), interpolation=cv2.INTER_LINEAR)
    img_train = background_bgr / 255.0
    label_train = [data['x0'] / 800.0,
                   data['x1'] / 800.0,
                   data['x2'] / 800.0,
                   data['x3'] / 800.0,
                   data['y0'] / 800.0,
                   data['y1'] / 800.0,
                   data['y2'] / 800.0,
                   data['y3'] / 800.0]
    return img_train, label_train


def get_norm_and_bgr_image_and_keypoint(exp_code: str):
    data, background_rgba = noise.get_noised_mtg_in_background(exp_code, card.get_illustration_id_random(exp_code))

    background_bgr = cv2.cvtColor(background_rgba, cv2.COLOR_BGRA2BGR)
    background_bgr = cv2.resize(background_bgr, (400, 400), interpolation=cv2.INTER_LINEAR)

    background_gray = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    img_train = background_gray / 255.0
    label_train = [data['x0'] / 800.0,
                   data['x1'] / 800.0,
                   data['x2'] / 800.0,
                   data['x3'] / 800.0,
                   data['y0'] / 800.0,
                   data['y1'] / 800.0,
                   data['y2'] / 800.0,
                   data['y3'] / 800.0]
    return img_train, label_train, background_bgr

if __name__ == '__main__':
    generate_data_set(10)
    print("pass")
