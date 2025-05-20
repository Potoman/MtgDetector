import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
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
    mtg = crop.get_rgba_image(exp_code, illustration_id)
    w = 488
    h = 680
    x0 = random.randint(0, 50)
    y0 = random.randint(0, 50)
    x1 = w - random.randint(0, 50)
    y1 = random.randint(0, 50)
    x2 = w - random.randint(0, 50)
    y2 = h - random.randint(0, 50)
    x3 = random.randint(0, 50)
    y3 = h - random.randint(0, 50)
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
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGBA)
    data, mtg = get_noised_mtg(exp_code, illustration_id)
    offset_x = random.randint((800 - 488) // 4, ((800 - 488) * 3) // 4)
    offset_y = random.randint((800 - 680) // 4, ((800 - 680) * 3) // 4)
    roi = background[offset_y:offset_y + 680, offset_x:offset_x + 488]
    mtg_mask = mtg[:, :, 3] != 0
    roi[mtg_mask] = mtg[mtg_mask]
    background[offset_y:offset_y + 680, offset_x:offset_x + 488] = roi
    data['x0'] = offset_x + data['x0']
    data['x1'] = offset_x + data['x1']
    data['x2'] = offset_x + data['x2']
    data['x3'] = offset_x + data['x3']
    data['y0'] = offset_y + data['y0']
    data['y1'] = offset_y + data['y1']
    data['y2'] = offset_y + data['y2']
    data['y3'] = offset_y + data['y3']
    return data, background


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
        coord, img = get_noised_mtg_in_background('mrd', 'ef02f536-d59d-4f80-a069-304c4d1bcc28')
        import overlay
        red = overlay.add_card_border(coord, img)
        cv2.imshow("mtg noised", red)
        mtg = noised_mtg_in_background_to_mtg(coord, img)
        cv2.imshow("mtg unoised", mtg)
        cv2.waitKey(0)

