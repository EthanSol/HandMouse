import joblib
import cv2
from HandDetector import HandDetector
from mediapipe_common import convert_landmarks_and_handedness_to_features, convert_hand_metadata_to_distances
import mediapipe as mp

class GestureDetector:
    def __init__(self, model_path="gesture_model.pkl", confidence_threshold=0.7):
        self.hand_detector = HandDetector()
        self.gesture_classifier = joblib.load(model_path)
        self.mp_drawing = mp.solutions.drawing_utils
        self._confidence_threshold = confidence_threshold

    def get_prediction_and_confidence(self, features):
        probs = self.gesture_classifier.predict_proba([features])[0]
        max_idx = probs.argmax()
        confidence = probs[max_idx]
        pred_class = self.gesture_classifier.classes_[max_idx]
        return pred_class, confidence

    def get_gesture(self, frame, debug=False):
        # Analyze frame for hand features
        hand_detection = self.hand_detector.getHandsFromRGBFrame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gesture_prediction = None
        wrist_pos = (0, 0, 0)
        if hand_detection.multi_hand_landmarks and hand_detection.multi_handedness:
            for hand_landmarks, handedness in zip(hand_detection.multi_hand_landmarks, hand_detection.multi_handedness):
                features = convert_hand_metadata_to_distances(hand_landmarks, handedness)

                prediction, confidence = self.get_prediction_and_confidence(features)
                if confidence >= self._confidence_threshold:
                    gesture_prediction = prediction

                wrist = hand_landmarks.landmark[0]
                wrist_pos = (wrist.x, wrist.y, wrist.z)

                if debug:
                    text = f"{prediction} ({confidence:.2f})" if gesture_prediction else f"Low conf ({confidence:.2f})"
                    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    wrist_text = f"Wrist: x={wrist.x:.2f}, y={wrist.y:.2f}, z={wrist.z:.2f}"
                    cv2.putText(frame, wrist_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.hand_detector.getHandConnections())

        return gesture_prediction, wrist_pos

    def close(self):
        self.hand_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
