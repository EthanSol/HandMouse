# parameters are expected to be outputs from a MediaPipe hand detection model.
def convert_landmarks_and_handedness_to_features(landmarks, handedness):
    # Flatten the landmarks
    features = []
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    
    # Add handedness feature
    left_right_feature = 0 if handedness.classification[0].label == 'Left' else 1
    features.append(left_right_feature)
    
    return features