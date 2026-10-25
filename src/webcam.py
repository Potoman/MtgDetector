import cv2
import algo_corner_detection
import overlay

if __name__ == '__main__':
    use_yolo = algo_corner_detection.selection_model == 5

    if use_yolo:
        model = algo_corner_detection.load_yolo_pose_model()
    else:
        model = algo_corner_detection.load_border_detection_model()

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, image_bgr = cap.read()

        # If frame not read correctly, break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        image_bgr_cropped = image_bgr[:, 80:(80 + 480), :]
        image_bgr_resized = cv2.resize(image_bgr_cropped, (400, 400), interpolation=cv2.INTER_LINEAR)

        if use_yolo:
            prediction = algo_corner_detection.get_yolo_prediction(model, image_bgr_resized)
        else:
            image_gray = cv2.cvtColor(image_bgr_resized, cv2.COLOR_BGR2GRAY)
            image_norm = image_gray / 255.0
            prediction = algo_corner_detection.get_prediction(model, image_norm)

        red = overlay.add_card_border(prediction, image_bgr_resized)

        cv2.imshow('Original', image_bgr)
        cv2.imshow('Resized', image_bgr_resized)
        cv2.imshow('Red', red)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()