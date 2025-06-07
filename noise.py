import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import card
import crop
import os
from typing import Dict

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, amount=0.02, salt_vs_pepper=0.5):
    noisy = image.copy()
    total_pixels = image.size

    # Salt noise
    num_salt = int(np.ceil(amount * total_pixels * salt_vs_pepper))
    coords = tuple(np.random.randint(0, i, num_salt) for i in image.shape[:2])
    noisy[coords] = 255

    # Pepper noise
    num_pepper = int(np.ceil(amount * total_pixels * (1.0 - salt_vs_pepper)))
    coords = tuple(np.random.randint(0, i, num_pepper) for i in image.shape[:2])
    noisy[coords] = 0

    return noisy

def add_poisson_noise(image):
    noisy = np.random.poisson(image.astype(np.float32))
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_speckle_noise(image):
    noise = np.random.randn(*image.shape) * 0.2
    noisy = image + image * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)



# # Create a blank image with a gray rectangle (simulate a card)
# image = np.ones((300, 400), dtype=np.uint8) * 150
# cv2.rectangle(image, (100, 80), (300, 220), (200), -1)  # A gray "card"
#
# # Apply various noise
# gaussian = add_gaussian_noise(image)
# salt_pepper = add_salt_pepper_noise(image)
# poisson = add_poisson_noise(image)
# speckle = add_speckle_noise(image)
#
# # Plot all images
# titles = ['Original', 'Gaussian', 'Salt & Pepper', 'Poisson', 'Speckle']
# images = [image, gaussian, salt_pepper, poisson, speckle]

# plt.figure(figsize=(15, 6))
# for i in range(5):
#     plt.subplot(1, 5, i+1)
#     plt.imshow(images[i], cmap='gray')
#     plt.title(titles[i])
#     plt.axis('off')
# plt.tight_layout()
# plt.show()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Create a blank canvas
# height, width = 300, 400
# image = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Generate wavy color noise using sine patterns
# for y in range(height):
#     for x in range(width):
#         r = 127 + 127 * np.sin(2 * np.pi * x / 60 + y / 30)
#         g = 127 + 127 * np.sin(2 * np.pi * y / 40 + x / 20)
#         b = 127 + 127 * np.sin(2 * np.pi * (x + y) / 80)
#
#         image[y, x] = [b, g, r]  # OpenCV uses BGR


def add_color_wave(image):
    color = image.copy()
    # Generate wavy color noise using sine patterns
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r = 127 + 127 * np.sin(2 * np.pi * x / 60 + y / 30)
            g = 127 + 127 * np.sin(2 * np.pi * y / 40 + x / 20)
            b = 127 + 127 * np.sin(2 * np.pi * (x + y) / 80)

            color[y, x] = [b, g, r]  # OpenCV uses BGR
    return color


# height, width = 300, 400
# image = np.zeros((height, width, 3), dtype=np.uint8)
# image_colored = add_color_wave(image)
#
# titles.append('colored')
# images.append(image_colored)

# # Show the result
# cv2.imshow("Wavy Color Noise", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def add_line_artifacts(image, num_lines=20, intensity=50):
    noisy = image.copy()
    h, w = noisy.shape[:2]

    for _ in range(num_lines):
        y = random.randint(0, h - 1)
        thickness = random.randint(1, 3)
        color = [random.randint(150, 255) for _ in range(3)]
        cv2.line(noisy, (0, y), (w, y), color, thickness)

        # Optional: vertical lines
        if random.random() > 0.7:
            x = random.randint(0, w - 1)
            cv2.line(noisy, (x, 0), (x, h), color, 1)

    return noisy

def add_block_artifacts(image, block_size=16):
    noisy = image.copy()
    h, w = noisy.shape[:2]

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            offset = random.randint(0, 20)
            noisy[y:y+block_size, x:x+block_size] = np.clip(
                noisy[y:y+block_size, x:x+block_size] + offset, 0, 255
            )
    return noisy

def add_scanlines(image, strength=50):
    noisy = image.copy()
    for i in range(0, noisy.shape[0], 2):
        noisy[i] = np.clip(noisy[i] - strength, 0, 255)
    return noisy


def show_noise():
    # Create a blank image with a gray rectangle (simulate a card)
    image = np.ones((300, 400), dtype=np.uint8) * 150
    cv2.rectangle(image, (100, 80), (300, 220), (200), -1)  # A gray "card"

    # Apply various noise
    gaussian = add_gaussian_noise(image)
    salt_pepper = add_salt_pepper_noise(image)
    poisson = add_poisson_noise(image)
    speckle = add_speckle_noise(image)

    # Plot all images
    titles = ['Original', 'Gaussian', 'Salt & Pepper', 'Poisson', 'Speckle']
    images = [image, gaussian, salt_pepper, poisson, speckle]

    height, width = 300, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image_colored = add_color_wave(image)

    titles.append('colored')
    images.append(image_colored)

    img = np.ones((300, 400, 3), dtype=np.uint8) * 128  # base gray canvas

    img = add_line_artifacts(img, num_lines=25)
    img = add_block_artifacts(img)
    img = add_scanlines(img)

    titles.append('artifacted')
    images.append(img)

    # cv2.imshow("Artifact Noise", img)

    plt.figure(figsize=(15, 6))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_noised_mtg(exp_code: str, illustration_id: str) -> (Dict[str, int], cv2.typing.MatLike):
    mtg = crop.get_bgra_image(exp_code, illustration_id)
    w = 488
    h = 680
    # Add distorsion :
    pixel_distorsion = 50
    x0 = int(random.randint(0, pixel_distorsion))
    y0 = int(random.randint(0, pixel_distorsion))
    x1 = w - int(random.randint(0, pixel_distorsion))
    y1 = int(random.randint(0, pixel_distorsion))
    x2 = w - int(random.randint(0, pixel_distorsion))
    y2 = h - int(random.randint(0, pixel_distorsion))
    x3 = int(random.randint(0, pixel_distorsion))
    y3 = h - int(random.randint(0, pixel_distorsion))
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32([[x0, y0], [x1, y1], [x3, y3], [x2, y2]])
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    twisted_image = cv2.warpPerspective(mtg, m, (w, h))
    return {'x0': x0, 'x1': x1, 'x2': x2, 'x3': x3, 'y0': y0, 'y1': y1, 'y2': y2, 'y3': y3}, twisted_image


def get_noised_mtg_in_background(exp_code: str, illustration_id: str) -> (Dict[str, int], cv2.typing.MatLike):
    backgrounds = {'Bathroom': ('bath', 1300),
                   'Bedroom': ('bed', 1432),
                   'Dinning': ('din', 1593),
                   'Kitchen': ('kitchen', 1360),
                   'Livingroom': ('living', 1427)}
    background = None
    while background is None:
        folder = random.choice(list(backgrounds.keys()))
        sub_image = backgrounds[folder]
        background_path = os.path.join('archive(6)', 'House_Room_Dataset', folder,
                                       sub_image[0] + '_' + str(random.randint(1, sub_image[1])) + '.jpg')
        if os.path.exists(background_path):
            background = cv2.imread(background_path)
    background = cv2.resize(background, (800, 800), interpolation=cv2.INTER_LINEAR)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    # Add blur :
    kernel = random.choice(range(5, 12, 2))
    background = cv2.GaussianBlur(background, (kernel, kernel), 0)

    data, mtg = get_noised_mtg(exp_code, illustration_id)
    # Add blur :
    kernel = random.choice(range(1, 12, 2))
    mtg = cv2.GaussianBlur(mtg, (kernel, kernel), 0)

    # Zoom here :
    zoom = 0.4 + random.random() * 0.6  # Random value from 0.4 to 1.0.
    new_width = int(488.0 * zoom)
    new_height = int(680.0 * zoom)
    mtg = cv2.resize(mtg, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    data['x0'] = int(data['x0'] * zoom)
    data['x1'] = int(data['x1'] * zoom)
    data['x2'] = int(data['x2'] * zoom)
    data['x3'] = int(data['x3'] * zoom)
    data['y0'] = int(data['y0'] * zoom)
    data['y1'] = int(data['y1'] * zoom)
    data['y2'] = int(data['y2'] * zoom)
    data['y3'] = int(data['y3'] * zoom)

    # Place the card randomly in the background with a margin of 25 mixel :
    # Margin of 25px where the card will not be present.
    margin = 15
    nb_pixel = (800 - (margin * 2) - new_width) * (800 - (margin * 2) - new_height)
    # Do it in uniform way by picking one random value
    pixel_position = random.randint(0, nb_pixel)
    offset_y = int(float(pixel_position) / (800 - (margin * 2) - new_width))
    offset_x = pixel_position - offset_y * (800 - (margin * 2) - new_width)
    # Add the margin
    offset_x = offset_x + margin
    offset_y = offset_y + margin
    # Appli the card
    roi = background[offset_y:offset_y + new_height, offset_x:offset_x + new_width]
    mtg_mask = mtg[:, :, 3] != 0
    roi[mtg_mask] = mtg[mtg_mask]
    background[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = roi
    data['x0'] = offset_x + data['x0']
    data['x1'] = offset_x + data['x1']
    data['x2'] = offset_x + data['x2']
    data['x3'] = offset_x + data['x3']
    data['y0'] = offset_y + data['y0']
    data['y1'] = offset_y + data['y1']
    data['y2'] = offset_y + data['y2']
    data['y3'] = offset_y + data['y3']
    background_bgr = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)


    # Add brightness :
    percent_brightness = random.randint(30, 70)
    if percent_brightness < 50:
        alpha = percent_brightness / 50.0
        beta = 0.0
    else:
        alpha = 1.0
        beta = 255.0 * (percent_brightness - 50) / 50.0
    background_bgr = cv2.convertScaleAbs(background_bgr, alpha=alpha, beta=beta)
    return data, background_bgr


def noised_mtg_in_background_to_mtg(data: Dict[str, int], img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    x0 = data['x0']
    x1 = data['x1']
    x2 = data['x2']
    x3 = data['x3']
    y0 = data['y0']
    y1 = data['y1']
    y2 = data['y2']
    y3 = data['y3']
    dst_pts = np.float32([[x0, y0], [x1, y1], [x3, y3], [x2, y2]])
    src_pts = np.float32([[0, 0], [488, 0], [0, 680], [488, 680]])
    m = cv2.getPerspectiveTransform(dst_pts, src_pts)
    mtg = cv2.warpPerspective(img, m, (488, 680))
    return mtg


if __name__ == '__main__':
    for a in range(100):
        coord, img = get_noised_mtg_in_background('mrd', card.get_illustration_id_random('mrd'))
        import overlay
        red = overlay.add_card_border(coord, img)
        cv2.imshow("mtg noised", red)
        mtg = noised_mtg_in_background_to_mtg(coord, img)
        cv2.imshow("mtg unoised", mtg)
        cv2.waitKey(0)

