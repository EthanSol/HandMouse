# HandMouse

HandMouse is a computer vision project that tracks hand movements using a camera and translates them into mouse input for controlling a cursor.

## How to Use HandMouse

1. Follow getting started below to set up the mouse.
2. Start with an open hand and move it in front of the camera.
3. The following hand gestures control the mouse:
   - **Open hand**: No action (cursor does not move or click).
   - **Pointing up and moving your hand**: Moves (grabs) the cursor.
   - **Fist**: Performs a left click.
   - **Shaka 🤙**: Performs a right click.

## Getting Started

### Optional: Automated Setup

The provided install.py script automates virtual environment and dependency setup. Skip to step 3 below after installing.

```bash
python install.py
```


### 1. (Recommended) Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Gesture Model

```bash
python train_gesture_model.py
```

### 4. Run the Project

```bash
python main.py
```

### Updating or Changing Gestures

The 4 gestures are represented by data files under the `gesture_data` folder. To update or change a gesture:

1. Delete the corresponding gesture data file in the `gesture_data` folder (e.g., `open_hand_data.csv`, `pointer_hand_data.csv`, `primary_select_hand_data.csv`, or `secondary_select_hand_data.csv`).
2. Run the following command:

```bash
python generateGestureTrainingData.py
```

3. When prompted, enter the gesture you want to replace (`open`, `pointer`, `primary_select`, or `secondary_select`).
4. When the camera opens, show the camera the gesture and hold the space bar to capture hand data.
5. Move your hand around the camera's field of view to capture multiple angles and perspectives. Move it closer and further away. Use both hands for more robust data.
6. When finished, close the window by pressing 'q'.
7. Retrain the model by running:

```bash
python train_gesture_model.py
```

Your new gesture data will now be used by HandMouse.

