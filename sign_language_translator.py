import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import pyttsx3

# Load trained model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

gesture_name = ""
last_gesture = ""
sentence = []

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

        if len(landmarks) == 42:
            landmarks_df = pd.DataFrame([landmarks])
            prediction = model.predict(landmarks_df)
            gesture_name = prediction[0]

            if gesture_name != last_gesture and gesture_name != "":
                print("Predicted gesture:", gesture_name)

                if gesture_name == "END":
                    # Speak the sentence if any words exist
                    sentence_text = " ".join(sentence)
                    if sentence_text.strip():
                        print("Speaking sentence:", sentence_text)
                        engine.say(sentence_text)
                        engine.runAndWait()
                        sentence = []  # Clear after speaking
                else:
                    sentence.append(gesture_name)

                last_gesture = gesture_name
    else:
        gesture_name = ""
        last_gesture = ""

    # Display the sentence so far
    sentence_text = " ".join(sentence)
    if sentence_text:
        cv2.putText(frame, f"Sentence: {sentence_text}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        # Show a placeholder when sentence is empty
        cv2.putText(frame, f"Sentence: ", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display current gesture
    if gesture_name:
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Clear gesture display if no hand visible
        cv2.putText(frame, "Gesture: ", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
