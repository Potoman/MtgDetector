import gc
import time

import tensorflow as tf
#from tf.keras import layers, models
import dataset

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def build_card_corner_model(input_shape=(400, 400, 1)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation='sigmoid')  # 1 for card presence and 4 corner points (x1, y1, ..., x4, y4)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# ===== 2. Custom Loss =====
def custom_loss(y_true, y_pred):
    y_true_coords = y_true[:, 1:]
    y_pred_coords = y_pred[:, 1:]

    y_true_presence = y_true[:, 0]
    y_pred_presence = y_pred[:, 0]

    keypoint_loss = tf.reduce_mean(tf.square(y_true_coords - y_pred_coords), axis=1)
    presence_loss = tf.keras.losses.binary_crossentropy(y_true_presence, y_pred_presence)

    return keypoint_loss + 0.5 * presence_loss


def generate_h5():
    model = build_card_corner_model()
    #model.compile(optimizer='adam', loss=custom_loss, metrics=['mae'])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    class StopAtAccuracy(tf.keras.callbacks.Callback):
        def __init__(self, target_acc: float):
            super().__init__()
            self.target_acc = target_acc

        def on_epoch_end(self, epoch, logs=None):
            current_acc = logs.get("mse")  # or "val_accuracy" for validation
            # print(f"\n🚨 Current accuracy {current_acc}...")
            # if current_acc is not None and current_acc >= self.target_acc:
            #     print(f"\n🚨 Target accuracy {self.target_acc} reached. Stopping training.")
            #     self.model.stop_training = True

    start_all = time.time()
    for step in range(10):
        print("Step... (" + str(step) + ")")
        start_step = time.time()

        print("Prepare dataset...")
        start = time.time()
        ds = dataset.generate_gray_dataset("mrd", 500)
        end = time.time()
        print("Prepare dataset : " + str(end - start))

        # print("Suffle...")
        # start = time.time()
        ds = ds.batch(4) #.prefetch(tf.data.AUTOTUNE)
        # end = time.time()
        # print("Suffle : " + str(end - start))

        print("Fit...")
        start = time.time()
        model.fit(ds, epochs=20, callbacks=[StopAtAccuracy(0.97)])
        end = time.time()
        print("Fit : " + str(end - start))

        print("Step : (" + str(step) + ")")

        del ds
        gc.collect()
        stop_step = time.time()
        print("Step time : (" + str(stop_step - start_step) + ")")

    end_all = time.time()
    print("All time : (" + str(end_all - start_all) + ")")

    model.save('model_border_detector.h5')

    return model


def predict_card(model):
    norm_image, keypoints, bgr_image = dataset.get_norm_and_bgr_image_and_keypoint('mrd')
    print(keypoints)

    input_tensor = tf.expand_dims(norm_image, axis=0)
    prediction = model.predict(input_tensor)[0]  # Shape: (9,)
    print(prediction)

    data = {}
    data['x0'] = int(prediction[0] * 400)
    data['x1'] = int(prediction[1] * 400)
    data['x2'] = int(prediction[2] * 400)
    data['x3'] = int(prediction[3] * 400)
    data['y0'] = int(prediction[4] * 400)
    data['y1'] = int(prediction[5] * 400)
    data['y2'] = int(prediction[6] * 400)
    data['y3'] = int(prediction[7] * 400)
    import overlay
    red = overlay.add_card_border(data, bgr_image)
    import cv2
    cv2.imwrite('input_image.png', bgr_image)
    cv2.imwrite('output_image.png', red)


    # corners_norm = prediction[1:]
    # presence = prediction[0]
    # img_bgr = image * 255.0
    # data = {}
    # data['x0'] = prediction[0] * 400
    # data['x1'] = prediction[1] * 400
    # data['x2'] = prediction[2] * 400
    # data['x3'] = prediction[3] * 400
    # data['y0'] = prediction[4] * 400
    # data['y1'] = prediction[5] * 400
    # data['y2'] = prediction[6] * 400
    # data['y3'] = prediction[7] * 400
    # import overlay
    # red = overlay.add_card_border(data, img_bgr)
    pass


def load_border_detection_model():
    model = tf.keras.models.load_model('model_border_detector.h5')
    return model

if __name__ == '__main__':
    #generate_h5()
    model = load_border_detection_model()
    predict_card(model)


# 100/100 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0335 - mae: 0.1664
# 🚨 Current accuracy None...
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 19ms/step - loss: 0.0335 - mae: 0.1664
# Fit : 39.22718358039856
# Step : (99)
# [0.24, 0.775, 0.77, 0.2075, 0.075, 0.06, 0.86875, 0.8775]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 642ms/step
# [0. 1. 1. 0. 0. 0. 1. 1.]
# (venvwsl) potoman@DESKTOP-MIVQCKB:/mnt/c/Documents and Settings/Maxime/Documents/work/mtg$


