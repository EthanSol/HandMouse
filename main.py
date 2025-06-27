import joblib
import cv2
import mediapipe as mp
from GestureDetector import GestureDetector
from mediapipe_common import convert_landmarks_and_handedness_to_features

# Initialize MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils

def main():
    gesture_classifier = joblib.load("gesture_model.pkl")

    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Use GestureDetector as a context manager
    with GestureDetector(model_path="gesture_model.pkl", confidence_threshold=0.0) as gesture_detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)

            gesture = gesture_detector.get_gesture(frame, debug=True)

            # Display the resulting frame
            cv2.imshow('Hand Tracking', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
