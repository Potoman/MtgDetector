import json
import cv2
import tensorflow as tf
import numpy as np
import crop


def train_exp():

    # Parameters
    image_size = (80, 80)
    num_classes = 3
    channels = 1  # Set to 3 for RGB
    batch_size = 64
    epochs = 30

    map_labels = {'': 0,
                  'znr': 1,
                  'mrd': 2}

    with open('labels.json', 'r') as file:
        labels = json.load(file)

        imgs_train = np.zeros((len(labels), 80, 80, 3), dtype=np.float32)
        labels_train = np.zeros((len(labels), 1), dtype=np.int32)

        index = 0
        for img_path, label in labels.items():
            img = tf.keras.utils.load_img(img_path)
            img_array = tf.keras.utils.img_to_array(img)
            #img_array = img_array.reshape(-1, 3)
            img_array = img_array / 255.0
            imgs_train[index] = img_array

            labels_train[index] = map_labels[label['exp_code']]

            index = index + 1


    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((imgs_train, labels_train))

    data_augmentation: tf.keras.Sequential = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.2),
    ])

    def augment(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32)  # Ensure float for augmentations
        image = data_augmentation(image, training=True)
        return image, label

    augmented_ds: tf.data.Dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    augmented_ds = augmented_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

    # Model
    model = tf.keras.Sequential([
        #tf.keras.layers.Input(shape=(*image_size, channels)),

        tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(80, 80, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=2),  # 20x20

        tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),  # 10x10

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Summary
    model.summary()

    class StopAtAccuracy(tf.keras.callbacks.Callback):
        def __init__(self, target_acc: float):
            super().__init__()
            self.target_acc = target_acc

        def on_epoch_end(self, epoch, logs=None):
            current_acc = logs.get("accuracy")  # or "val_accuracy" for validation
            if current_acc is not None and current_acc >= self.target_acc:
                print(f"\n🚨 Target accuracy {self.target_acc} reached. Stopping training.")
                self.model.stop_training = True

    # Train
    model.fit(augmented_ds,
              epochs=epochs,
              callbacks=[StopAtAccuracy(0.97)])

    crop_and_labels = crop.get_crop_and_labels('mrd')
    print(f"Card in mrd : {len(crop_and_labels)}")

    imgs_train = np.zeros((len(crop_and_labels), 80, 80, 3), dtype=np.float32)
    for index in range(len(crop_and_labels)):
        imgs_train[index] = crop_and_labels[index][0]

    mrd_result = model(imgs_train)


    crop_and_labels = crop.get_crop_and_labels('znr')
    print(f"Card in znr : {len(crop_and_labels)}")

    imgs_train = np.zeros((len(crop_and_labels), 80, 80, 3), dtype=np.float32)
    for index in range(len(crop_and_labels)):
        imgs_train[index] = crop_and_labels[index][0]

    znr_result = model(imgs_train)



def pass_canny(image):
    original = image.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges with Canny
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=1*np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

    if lines is None:
        return original

    for i in lines:
        x1, x2, y1, y2 = i[0]
        cv2.line(original, (x1, x2), (y1, y2), (0, 255, 0), 3)

    return original


if __name__ == '__main__':
    train_exp()
    pass