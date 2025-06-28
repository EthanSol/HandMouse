import joblib
import cv2
import mediapipe as mp
from GestureDetector import GestureDetector
from CursorController import CursorController, CursorAction
from mediapipe_common import convert_landmarks_and_handedness_to_features

# Initialize MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils

def gesture_to_cursor_action(gesture):
    if gesture == "primary_select":
        return CursorAction.LeftClick
    elif gesture == "secondary_select":
        return CursorAction.RightClick
    elif gesture == "pointer":
        return CursorAction.MoveCursor
    else:
        return CursorAction.NoAction

def main():
    gesture_classifier = joblib.load("gesture_model.pkl")

    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    cursor_controller = CursorController()

    # Use GestureDetector as a context manager
    with GestureDetector(model_path="gesture_model.pkl", confidence_threshold=0.4) as gesture_detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)

            gesture, wrist_pos = gesture_detector.get_gesture(frame, debug=True)

            cursor_action = gesture_to_cursor_action(gesture)
            cursor_controller.update_cursor(cursor_action, wrist_pos[0], wrist_pos[1])

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
