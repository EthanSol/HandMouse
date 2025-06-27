# This script collects gesture training data using MediaPipe Hands and saves it to a CSV file.
# This data will be used to train a model for gesture recognition.

import csv
import cv2
import mediapipe as mp
import os
from mediapipe_common import convert_landmarks_and_handedness_to_features

# Initialize MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

label = input("Enter gesture label (e.g., 'fist'): ")
data = []

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.waitKey(1) & 0xFF == ord(' '):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                row = convert_landmarks_and_handedness_to_features(hand_landmarks, handedness)
                row.append(label)
                data.append(row)

                mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
    cv2.imshow("Collecting", frame)

# Save to CSV
os.makedirs('gesture_data', exist_ok=True)
csv_path = f'gesture_data/{label}_hand_data.csv'
with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

cap.release()
cv2.destroyAllWindows()
