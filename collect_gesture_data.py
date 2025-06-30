import cv2
import mediapipe as mp
import numpy as np
import csv
from collections import defaultdict

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

gesture_label = None

# Open CSV file for appending
file = open("gesture_data.csv", mode="a", newline='')
writer = csv.writer(file)

sample_counts = defaultdict(int)
feedback_message = ""

print("Press ENTER to change gesture label, or q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                landmarks.extend([x, y])

    # Show current label
    label_text = f"Current Label: {gesture_label}" if gesture_label else "No Label Selected"
    cv2.putText(frame, label_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show sample counts
    if gesture_label:
        count_text = f"Samples Collected: {sample_counts[gesture_label]}"
        cv2.putText(frame, count_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if feedback_message:
        cv2.putText(frame, feedback_message, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collect Gesture Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('\r') or key == 13:  # Enter key
        gesture_label = input("Enter new gesture label: ").strip().upper()
        feedback_message = f"Gesture label set to {gesture_label}"
    elif key == ord('s'):
        if gesture_label and landmarks and len(landmarks) == 42:
            row = landmarks + [gesture_label]
            writer.writerow(row)
            sample_counts[gesture_label] += 1
            feedback_message = f"Sample saved for {gesture_label}"
        else:
            feedback_message = "No hand detected or no label set."
    elif key == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
