import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import crop

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


def insert_card():
    image = np.zeros((800, 800, 4), dtype=np.uint8)
    cv2.imshow("black image", image)

    mtg = crop.get_rgba_image('mrd', 'ef02f536-d59d-4f80-a069-304c4d1bcc28')
    cv2.imshow("mtg", mtg)

    image[60:60+680, 156:156+488] = mtg
    cv2.imshow("mtg in white", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    insert_card()

