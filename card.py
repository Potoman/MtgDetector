from ultralytics import YOLO
import cv2
import algo
import crop
import json
import random
from typing import Dict, List


data_base: Dict[str, List[str]] = {}


def get_illustrations_ids(exp_code: str) -> List[str]:
    if exp_code not in data_base:
        illustration_ids = []
        with open(f'result_{exp_code}.json', 'r') as file:
            cards = json.load(file)
            for card in cards:
                # Some cards have multiple faces
                if "image_uris" in card:
                    # Single-faced card
                    illustration_ids.append(f"{card['illustration_id']}")
                elif "card_faces" in card:
                    # Double-faced card
                    for i, face in enumerate(card["card_faces"]):
                        if "image_uris" in face:
                            illustration_ids.append(f"{face['illustration_id']}")
        data_base[exp_code] = illustration_ids
    return data_base[exp_code]


def get_illustration_id_random(exp_code: str):
    return random.choice(get_illustrations_ids(exp_code))


def get_argb_random(exp_code: str):
    return crop.get_rgba_image(exp_code, get_illustration_id_random(exp_code))


def webcam_to_canny():
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame not read correctly, break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        image = algo.pass_canny(frame)
        # Display the resulting frame
        cv2.imshow('Canny', image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def webcam_to_yolo():
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame not read correctly, break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        # Display the resulting frame
        cv2.imshow('Webcam', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def webcam_to_find_contours():

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame not read correctly, break
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optional: blur and edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours and find a quadrilateral
        for cnt in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Check for 4-sided polygon with area threshold
            if len(approx) == 4 and cv2.contourArea(approx) > 1000:
                # Draw the contour (optional)
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

                # Extract corner points
                corners = approx.reshape(4, 2)
                print("Card corners:")
                print(corners)

                # Break after first card found
                break

        # Show result
        cv2.imshow('Card Corners', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    webcam_to_canny()
    pass