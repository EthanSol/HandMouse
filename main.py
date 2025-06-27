import cv2
import mediapipe as mp
from HandDetector import HandDetector

# Initialize MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils

def main():
    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Use HandDetector as a context manager
    with HandDetector() as hand_detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Flip the frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)

            results = hand_detector.getHandsFromRGBFrame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, hand_detector.getHandConnections())

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
