import cv2
import gc
import overlay
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
        tf.keras.layers.Dense(9, activation='sigmoid')  # 1 for card presence and 4 corner points (x1, y1, ..., x4, y4)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


mse_loss_fn = tf.keras.losses.MeanSquaredError(reduction='none')  # per-sample loss
bce_loss_fn = tf.keras.losses.BinaryCrossentropy()


# ===== 2. Custom Loss =====
def custom_loss(y_true, y_pred):
    # Extract values
    y_true_coords = y_true[:, :8]  # First 8 values: coordinates
    y_true_presence = y_true[:, 8]  # Last value: presence flag

    y_pred_coords = y_pred[:, :8]
    y_pred_presence = y_pred[:, 8]

    # Coordinate loss: Mean Squared Error
    coord_loss = mse_loss_fn(y_true_coords, y_pred_coords)

    # Apply mask: only consider coordinate loss if object is present
    coord_loss = coord_loss * y_true_presence

    # Presence loss: Binary Crossentropy
    presence_loss = bce_loss_fn(y_true_presence, y_pred_presence)

    # Combine both losses
    total_loss = coord_loss + presence_loss

    # Mean over batch
    return tf.reduce_mean(total_loss)


def presence_accuracy(y_true, y_pred):
    y_true_presence = y_true[:, 8]
    y_pred_presence = y_pred[:, 8]
    return tf.keras.metrics.binary_accuracy(y_true_presence, y_pred_presence)


def coord_mae_when_present(y_true, y_pred):
    y_true_coords = y_true[:, :8]
    y_pred_coords = y_pred[:, :8]
    y_true_presence = y_true[:, 8]

    abs_error = tf.abs(y_true_coords - y_pred_coords)
    mae_per_sample = tf.reduce_mean(abs_error, axis=1)

    # Mask with presence flag
    mae_masked = mae_per_sample * y_true_presence

    # Avoid division by 0 if no present objects in batch
    total_present = tf.reduce_sum(y_true_presence)
    return tf.reduce_sum(mae_masked) / (total_present + 1e-7)

def generate_h5():
    model = build_card_corner_model()
    #model.compile(optimizer='adam', loss=custom_loss, metrics=['mae'])
    model.compile(optimizer='adam', loss=custom_loss, metrics=[presence_accuracy, coord_mae_when_present])

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

    combined_history = {
        'coord_mae_when_present': [],
        'loss': [],
        'presence_accuracy': []
    }

    count_step = 2
    size_dataset = 100
    count_epoch = 1
    start_all = time.time()
    for epoch in range(count_epoch):
        for step in range(count_step):
            print(f"Epoch = {epoch}, Step = {step}...")
            start_step = time.time()

            print("Prepare dataset...")
            start = time.time()
            ds = dataset.generate_gray_dataset("mrd", step, size_dataset)
            end = time.time()
            print("Prepare dataset : " + str(end - start))

            # print("Suffle...")
            # start = time.time()
            ds = ds.batch(4) #.prefetch(tf.data.AUTOTUNE)
            # end = time.time()
            # print("Suffle : " + str(end - start))

            print("Fit...")
            start = time.time()
            history = model.fit(ds, epochs=1, callbacks=[StopAtAccuracy(0.97)])
            for key in combined_history:
                combined_history[key].extend(history.history.get(key, []))

            end = time.time()
            print("Fit : " + str(end - start))

            print("Step : (" + str(step) + ")")

            del ds
            gc.collect()
            stop_step = time.time()
            print("Step time : (" + str(stop_step - start_step) + ")")

    end_all = time.time()
    print("All time : (" + str(end_all - start_all) + ")")

    import matplotlib.pyplot as plt
    plt.plot(combined_history['coord_mae_when_present'], label='MAE')
    plt.plot(combined_history['loss'], label='Training loss')
    plt.legend()
    plt.title(f"step : {count_step}; data set size : {size_dataset}; epoch : {count_epoch}")
    plt.savefig(f"training_s{count_step}_ds{size_dataset}_e{count_epoch}.png")
    plt.close()

    model.save('model_border_detector.keras')

    return model


def predict_random_card(model):
    norm_image, keypoints, bgr_image = dataset.get_norm_and_bgr_image_and_keypoint('znr')
    print(keypoints)
    prediction = get_prediction(model, norm_image)
    red = overlay.add_card_border(prediction, bgr_image)
    cv2.imwrite('input_image.png', bgr_image)
    cv2.imwrite('output_image.png', red)


def get_prediction(model, image_norm):
    input_tensor = tf.expand_dims(image_norm, axis=0)
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
    data['is_present'] = 1.0 if prediction[8] > 0.5 else 0.0
    return data


def load_border_detection_model():
    model = tf.keras.models.load_model('model_border_detector.keras', custom_objects={"custom_loss": custom_loss, "presence_accuracy": presence_accuracy, "coord_mae_when_present": coord_mae_when_present})
    return model

if __name__ == '__main__':
    generate_h5()
    #model = load_border_detection_model()
    #predict_random_card(model)


# 100/100 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - loss: 0.0335 - mae: 0.1664
# 🚨 Current accuracy None...
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 2s 19ms/step - loss: 0.0335 - mae: 0.1664
# Fit : 39.22718358039856
# Step : (99)
# [0.24, 0.775, 0.77, 0.2075, 0.075, 0.06, 0.86875, 0.8775]
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 642ms/step
# [0. 1. 1. 0. 0. 0. 1. 1.]
# (venvwsl) potoman@DESKTOP-MIVQCKB:/mnt/c/Documents and Settings/Maxime/Documents/work/mtg$


