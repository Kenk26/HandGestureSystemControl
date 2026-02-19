"""
Configuration - Hand Gesture Controller
Architecture: MediaPipe Landmarks (63 features) → TensorFlow Dense Model → Mouse Actions
"""

import platform

# ── Gestures ─────────────────────────────────────────────────────────────────
GESTURES = {
    0: 'palm',
    1: 'index',
    2: 'peace',
    3: 'fist',
    4: 'thumb_pinky',
    5: 'okay'
}
NUM_CLASSES = len(GESTURES)

# ── Model / Feature Settings ──────────────────────────────────────────────────
# MediaPipe gives 21 landmarks x 3 coords (x, y, z) = 63 input features
NUM_LANDMARKS  = 21
NUM_COORDS     = 3
INPUT_FEATURES = NUM_LANDMARKS * NUM_COORDS   # 63

MODEL_PATH           = 'models/gesture_model.h5'
GESTURE_MAPPING_PATH = 'models/gesture_mapping.json'

# Minimum TF prediction confidence to accept a gesture (0-1)
CONFIDENCE_THRESHOLD = 0.85

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MEDIAPIPE_MODEL_PATH = 'hand_landmarker.task'
MEDIAPIPE_MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)
MAX_HANDS            = 1
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE  = 0.5

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

# ── Mouse / Action ────────────────────────────────────────────────────────────
MOUSE_SMOOTHING  = 5
MOVE_SENSITIVITY = 0.1    # multiplier for relative palm movement delta
ACTION_COOLDOWN  = 0.8    # seconds between repeated clicks
SCROLL_AMOUNT    = 30
SCROLL_THRESHOLD = 0.008  # min y-delta to trigger scroll

# ── Gesture Smoothing ─────────────────────────────────────────────────────────
GESTURE_HISTORY_LENGTH     = 6   # frames kept in ring buffer
GESTURE_STABILITY_REQUIRED = 4   # frames that must agree to confirm gesture

# ── Mini Overlay Window ───────────────────────────────────────────────────────
SHOW_MINI_WINDOW      = True
MINI_WIDTH            = 280
MINI_HEIGHT           = 210
MINI_BORDER_COLOR     = (0, 255, 0)    # BGR green
MINI_BORDER_THICKNESS = 2

# ── Gesture -> Action Mapping ──────────────────────────────────────────────────
GESTURE_ACTIONS = {
    'palm'       : 'move_cursor',
    'index'      : 'left_click',
    'peace'      : 'right_click',
    'fist'       : 'scroll',
    'thumb_pinky': 'double_click',
    'okay'       : 'hold_drag',
}

# ── App Launcher (customise per OS) ───────────────────────────────────────────
_sys = platform.system()
APP_MAPPINGS = {
    'Windows': {'app1': 'notepad',  'app2': 'calc.exe'},
    'Darwin' : {'app1': 'TextEdit', 'app2': 'Calculator'},
    'Linux'  : {'app1': 'gedit',    'app2': 'gnome-calculator'},
}.get(_sys, {'app1': 'notepad', 'app2': 'calc.exe'})

# ── Training ──────────────────────────────────────────────────────────────────
SAMPLES_PER_GESTURE = 300    # 300 x 6 gestures = 1800 total
TRAINING_EPOCHS     = 100
BATCH_SIZE          = 32
TEST_SIZE           = 0.2

# ── UI Colours (BGR) ──────────────────────────────────────────────────────────
GESTURE_COLORS = {
    'palm'       : (200, 200, 200),   # grey
    'index'      : (255, 255,   0),   # cyan
    'peace'      : (  0, 255,   0),   # green
    'fist'       : (  0,   0, 255),   # red
    'thumb_pinky': (  0, 165, 255),   # orange
    'okay'       : (  0, 255, 255),   # cyan  ← hold & drag
    'unknown'    : (160, 160, 160),
}

# ── Debug ─────────────────────────────────────────────────────────────────────
SHOW_FPS        = True
SHOW_CONFIDENCE = True
VERBOSE         = False