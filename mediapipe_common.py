def handedness_to_int(handedness):
    """
    Convert handedness classification to 0 (Left) or 1 (Right).
    """
    return 0 if handedness.classification[0].label == 'Left' else 1

# parameters are expected to be outputs from a MediaPipe hand detection model.
def convert_landmarks_and_handedness_to_features(landmarks, handedness):
    # Flatten the landmarks
    features = []
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    
    # Add handedness feature
    left_right_feature = handedness_to_int(handedness)
    features.append(left_right_feature)
    
    return features

# Feature engineering

# current transformations:
# - Finger base-to-tip distances (squared)
# - Wrist-to-finger base distances (squared)
# - Handedness (0=Left, 1=Right)

# possible future transformations for more complex gestures:
# - Finger angles
# - Distances from finger tips to each other or to wrist

# parameters are expected to be outputs from a MediaPipe hand detection model.
def convert_hand_metadata_to_distances(landmarks, handedness):
    """
    Returns a list of features: 5 base-to-tip distances, 5 wrist-to-base distances, and handedness (0=Left, 1=Right).
    """
    features = get_finger_base_to_tip_distances_sqr(landmarks)
    features += get_wrist_to_finger_base_distances(landmarks)
    features.append(handedness_to_int(handedness))
    return features

def get_finger_base_to_tip_distances_sqr(landmarks):
    """
    Calculate Manhattan distances from the base to the tip of each finger using hand landmarks.
    Returns a list of 5 distances (thumb, index, middle, ring, pinky).
    """
    finger_indices = [
        (1, 4),   # Thumb: base, tip
        (5, 8),   # Index: base, tip
        (9, 12),  # Middle: base, tip
        (13, 16), # Ring: base, tip
        (17, 20)  # Pinky: base, tip
    ]
    dists = []
    for base_idx, tip_idx in finger_indices:
        base = landmarks.landmark[base_idx]
        tip = landmarks.landmark[tip_idx]
        dx = abs(tip.x - base.x)
        dy = abs(tip.y - base.y)
        dz = abs(tip.z - base.z)
        dist = dx + dy + dz
        dists.append(dist)
    return dists

def get_wrist_to_finger_base_distances(landmarks):
    """
    Calculate Manhattan distances from the wrist to the base of each finger using hand landmarks.
    Returns a list of 5 distances (thumb, index, middle, ring, pinky).
    """
    wrist = landmarks.landmark[0]
    finger_bases = [1, 5, 9, 13, 17]
    dists = []
    for base_idx in finger_bases:
        base = landmarks.landmark[base_idx]
        dx = abs(base.x - wrist.x)
        dy = abs(base.y - wrist.y)
        dz = abs(base.z - wrist.z)
        dist = dx + dy + dz
        dists.append(dist)
    return dists