import mediapipe as mp

# Initialize MediaPipe Drawing Utils
mp_drawing = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def getHandsFromRGBFrame(self, rgb_frame):
       return self._hands.process(rgb_frame)

    def getHandConnections(self):
        return self._mp_hands.HAND_CONNECTIONS

    # Support for context management
    def close(self):
        self._hands.close()
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
