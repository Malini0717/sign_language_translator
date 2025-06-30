import cv2
import mediapipe as mp
import numpy as np
import csv

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Start video capture
cap = cv2.VideoCapture(0)

gesture_label = "HELLO"  # default label

# Open CSV file for appending
file = open("gesture_data.csv", mode="a", newline='')
writer = csv.writer(file)

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

    # Show the current label on screen
    cv2.putText(frame, f"Current Label: {gesture_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Collect Gesture Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('h'):
        gesture_label = "HELLO"
        print("Gesture label set to HELLO")
    elif key == ord('y'):
        gesture_label = "YES"
        print("Gesture label set to YES")
    elif key == ord('n'):
        gesture_label = "NO"
        print("Gesture label set to NO")
    elif key == ord('s'):
        if landmarks and len(landmarks) == 42:
            row = landmarks + [gesture_label]
            writer.writerow(row)
            print(f"Sample saved for gesture: {gesture_label}")
        else:
            print("No hand detected to save sample.")
    elif key == ord('q'):
        break

cap.release()
file.close()
cv2.destroyAllWindows()
