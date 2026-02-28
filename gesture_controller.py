"""
Hand Gesture Controller
Pipeline: MediaPipe landmarks -> TensorFlow model -> Mouse/System actions
Gestures: palm=move, index=left_click, peace=right_click, fist=scroll,
          thumb_pinky=double_click, okay=none, pinch=click+drag
"""

import cv2
import numpy as np
import time
import os
import json
import urllib.request
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import pyautogui

from mouse_controller import MouseController

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import (
        GESTURES, NUM_CLASSES, INPUT_FEATURES,
        MODEL_PATH, GESTURE_MAPPING_PATH, CONFIDENCE_THRESHOLD,
        MEDIAPIPE_MODEL_PATH, MEDIAPIPE_MODEL_URL,
        DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
        CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
        MOUSE_SMOOTHING, MOVE_SENSITIVITY, ACTION_COOLDOWN,
        SCROLL_AMOUNT, SCROLL_THRESHOLD,
        GESTURE_HISTORY_LENGTH, GESTURE_STABILITY_REQUIRED,
        SHOW_MINI_WINDOW, MINI_WIDTH, MINI_HEIGHT,
        MINI_BORDER_COLOR, MINI_BORDER_THICKNESS,
        GESTURE_ACTIONS, GESTURE_COLORS,
        SHOW_FPS, SHOW_CONFIDENCE,
    )
except ImportError:
    # Fallback defaults if config.py is missing
    GESTURES                  = {0:'palm',1:'index',2:'peace',3:'fist',4:'thumb_pinky',5:'okay'}
    NUM_CLASSES               = 6
    INPUT_FEATURES            = 63
    MODEL_PATH                = 'models/gesture_model.h5'
    GESTURE_MAPPING_PATH      = 'models/gesture_mapping.json'
    CONFIDENCE_THRESHOLD      = 0.85
    MEDIAPIPE_MODEL_PATH      = 'hand_landmarker.task'
    MEDIAPIPE_MODEL_URL       = ('https://storage.googleapis.com/mediapipe-models/'
                                 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
    DETECTION_CONFIDENCE      = 0.7
    TRACKING_CONFIDENCE       = 0.5
    CAMERA_INDEX              = 0
    CAMERA_WIDTH              = 640
    CAMERA_HEIGHT             = 480
    MOUSE_SMOOTHING           = 5
    MOVE_SENSITIVITY          = 0.1
    ACTION_COOLDOWN           = 0.8
    SCROLL_AMOUNT             = 30
    SCROLL_THRESHOLD          = 0.008
    GESTURE_HISTORY_LENGTH    = 6
    GESTURE_STABILITY_REQUIRED= 4
    SHOW_MINI_WINDOW          = True
    MINI_WIDTH                = 280
    MINI_HEIGHT               = 210
    MINI_BORDER_COLOR         = (0, 255, 0)
    MINI_BORDER_THICKNESS     = 2
    GESTURE_ACTIONS           = {'palm':'move_cursor','index':'left_click',
                                  'peace':'right_click','fist':'scroll',
                                  'thumb_pinky':'double_click','okay':'hold_drag'}
    GESTURE_COLORS            = {'palm':(200,200,200),'index':(255,255,0),
                                  'peace':(0,255,0),'fist':(0,0,255),
                                  'thumb_pinky':(0,165,255),'okay':(0,255,255),
                                  'unknown':(160,160,160)}
    SHOW_FPS                  = True
    SHOW_CONFIDENCE           = True


# ── Skeleton connections ──────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(5,17)
]

# ── Pinch detection (landmark-based, no TF model needed) ─────────────────────
# Thumb tip = 4, Index tip = 8. Normalised distance below this = pinch active.
PINCH_THRESHOLD = 0.07   # tune up/down if too sensitive or hard to trigger


# ── Download MediaPipe model if needed ────────────────────────────────────────
if not os.path.exists(MEDIAPIPE_MODEL_PATH):
    print("Downloading hand landmarker model (~13 MB)...")
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
    print("  Model downloaded")


# ── Landmark normalisation (must match collect_data.py exactly) ───────────────
def normalize_landmarks(lms):
    """
    Normalize 21 landmarks relative to wrist (landmark 0).
    Returns numpy array of shape (1, 63) ready for TF inference.
    """
    base_x, base_y, base_z = lms[0].x, lms[0].y, lms[0].z
    coords = []
    for lm in lms:
        coords.append(lm.x - base_x)
        coords.append(lm.y - base_y)
        coords.append(lm.z - base_z)
    max_val = max(abs(v) for v in coords) or 1.0
    coords  = [v / max_val for v in coords]
    return np.array(coords, dtype=np.float32).reshape(1, -1)


# ── GestureController ─────────────────────────────────────────────────────────
class GestureController:

    def __init__(self):
        self._load_tf_model()
        self._init_mediapipe()
        self.mouse      = MouseController(smoothing=MOUSE_SMOOTHING)
        self.history    = deque(maxlen=GESTURE_HISTORY_LENGTH)
        self.last_action= {}
        self.prev_scroll_y = None
        self.prev_palm_x   = None
        self.prev_palm_y   = None
        self.fps_q      = deque(maxlen=10)
        self.prev_time  = time.time()
        self.confidence = 0.0
        self.last_gesture   = 'unknown'   # cache last TF result for skipped frames
        self.tf_frame_skip  = 0           # counter for running TF every N frames
        # ── Hold/drag state (okay gesture) ───────────────────────────────
        self.drag_active   = False
        self.drag_x        = None
        self.drag_y        = None
        # ── Warm up TF model (first call is always slow) ──────────────────
        dummy = np.zeros((1, INPUT_FEATURES), dtype=np.float32)
        self.tf_model(dummy, training=False)
        print("  TF model warmed up")
        self._print_startup()

    # ── init helpers ──────────────────────────────────────────────────────────

    def _load_tf_model(self):
        """Load the trained TensorFlow landmark classifier."""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "  Please train first:\n"
                "    1. python collect_data.py\n"
                "    2. Open train_model.ipynb and run all cells"
            )
        print(f"Loading TensorFlow model from {MODEL_PATH} ...")
        self.tf_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"  Model loaded  |  Input: {self.tf_model.input_shape}  "
              f"|  Output: {self.tf_model.output_shape}")

        # Load gesture label map
        if os.path.exists(GESTURE_MAPPING_PATH):
            with open(GESTURE_MAPPING_PATH) as f:
                raw = json.load(f)
            self.gesture_map = {int(k): v for k, v in raw.items()}
        else:
            self.gesture_map = GESTURES

    def _init_mediapipe(self):
        """Initialise MediaPipe Hand Landmarker."""
        base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )
        self.mp_detector = vision.HandLandmarker.create_from_options(options)

    def _print_startup(self):
        print("\n" + "="*58)
        print("  HAND GESTURE CONTROLLER  (TF + MediaPipe)")
        print("="*58)
        for g, a in GESTURE_ACTIONS.items():
            print(f"  {g:14s} -> {a}")
        print(f"  {'pinch':14s} -> click + drag  (landmark-based)")
        print("-"*58)
        print("  Q = quit    P = pause / resume")
        print("="*58 + "\n")

    # ── per-frame helpers ─────────────────────────────────────────────────────

    def _predict_gesture(self, lms):
        """
        Run TensorFlow inference on normalized landmarks.
        Uses direct model call (faster than model.predict on single samples).
        Returns (gesture_name, confidence) or ('unknown', conf) if below threshold.
        """
        features = normalize_landmarks(lms)          # shape (1, 63)
        probs    = self.tf_model(features, training=False).numpy()[0]
        top_id   = int(np.argmax(probs))
        top_conf = float(probs[top_id])

        self.confidence = top_conf
        if top_conf >= CONFIDENCE_THRESHOLD:
            return self.gesture_map.get(top_id, 'unknown'), top_conf
        return 'unknown', top_conf

    def _stable_gesture(self, g):
        """Return confirmed gesture only when enough frames agree."""
        self.history.append(g)
        if len(self.history) < self.history.maxlen:
            return None
        counts = {}
        for x in self.history:
            counts[x] = counts.get(x, 0) + 1
        best = max(counts, key=counts.get)
        return best if counts[best] >= GESTURE_STABILITY_REQUIRED else None

    def _can_act(self, key):
        """Cooldown guard — True only if enough time has passed."""
        now = time.time()
        if now - self.last_action.get(key, 0) >= ACTION_COOLDOWN:
            self.last_action[key] = now
            return True
        return False

    def _fps(self):
        now = time.time()
        self.fps_q.append(1 / max(now - self.prev_time, 1e-6))
        self.prev_time = now
        return sum(self.fps_q) / len(self.fps_q)

    # ── drawing helpers ───────────────────────────────────────────────────────

    def _draw_skeleton(self, frame, lms, color=(0, 220, 0)):
        h, w = frame.shape[:2]
        pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

    def _get_bbox(self, frame, lms, margin=25):
        h, w = frame.shape[:2]
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        x1 = max(0, int(min(xs) * w) - margin)
        y1 = max(0, int(min(ys) * h) - margin)
        x2 = min(w, int(max(xs) * w) + margin)
        y2 = min(h, int(max(ys) * h) + margin)
        return x1, y1, x2, y2

    def _draw_main_ui(self, frame, gesture, action_text, fps_val):
        h, w = frame.shape[:2]

        # Top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 58), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, "HAND GESTURE CONTROLLER",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)

        if SHOW_FPS:
            cv2.putText(frame, f"FPS {int(fps_val)}",
                        (w - 100, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2, cv2.LINE_AA)

        # Gesture badge
        if gesture and gesture != 'unknown':
            color = GESTURE_COLORS.get(gesture, (255, 255, 255))
            label = gesture.upper()
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (8, 66), (tw + 24, 98), color, -1)
            cv2.putText(frame, label, (14, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

            # Confidence bar
            if SHOW_CONFIDENCE:
                bar_len = int(200 * self.confidence)
                cv2.rectangle(frame, (tw + 34, 75), (tw + 34 + 200, 88), (50, 50, 50), -1)
                cv2.rectangle(frame, (tw + 34, 75), (tw + 34 + bar_len, 88), color, -1)
                cv2.putText(frame, f"{self.confidence*100:.0f}%",
                            (tw + 244, 88), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Action banner at bottom centre
        if action_text:
            (tw, th), _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            tx = (w - tw) // 2
            ty = h - 50
            cv2.rectangle(frame, (tx - 14, ty - th - 10), (tx + tw + 14, ty + 10), (0, 0, 0), -1)
            cv2.putText(frame, action_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)

        # Bottom hint
        cv2.putText(frame, "Q=quit  P=pause  M=mini window",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (140, 140, 140), 1, cv2.LINE_AA)

    def _build_mini_frame(self, frame, gesture):
        """
        Create the mini overlay: small camera view with skeleton + gesture label.
        Positioned in top-right corner of the screen.
        """
        mini = cv2.resize(frame, (MINI_WIDTH, MINI_HEIGHT))

        # Coloured border matching current gesture
        color = GESTURE_COLORS.get(gesture, MINI_BORDER_COLOR)
        cv2.rectangle(mini, (0, 0),
                      (MINI_WIDTH - 1, MINI_HEIGHT - 1),
                      color, MINI_BORDER_THICKNESS)

        # Gesture label inside mini window
        if gesture and gesture != 'unknown':
            label = gesture.upper()
            cv2.putText(mini, label,
                        (6, MINI_HEIGHT - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)

        return mini

    # ── action execution ──────────────────────────────────────────────────────

    def _execute(self, gesture, lms):
        """Map confirmed gesture to mouse/system action. Returns action label."""
        action = GESTURE_ACTIONS.get(gesture, '')

        if action == 'move_cursor':
            cur_x = lms[9].x
            cur_y = lms[9].y
            if self.prev_palm_x is not None:
                dx = (cur_x - self.prev_palm_x) * MOVE_SENSITIVITY
                dy = (cur_y - self.prev_palm_y) * MOVE_SENSITIVITY
                if abs(dx) > 0.002 or abs(dy) > 0.002:
                    cx, cy = self.mouse.get_cursor_position()
                    new_x  = max(0, min(self.mouse.screen_width  - 1,
                                        cx + int(dx * self.mouse.screen_width)))
                    new_y  = max(0, min(self.mouse.screen_height - 1,
                                        cy + int(dy * self.mouse.screen_height)))
                    pyautogui.moveTo(new_x, new_y)
                    return "Moving Cursor"
            self.prev_palm_x = cur_x
            self.prev_palm_y = cur_y
            return ""

        # Reset palm tracking for non-move gestures
        self.prev_palm_x = None
        self.prev_palm_y = None

        # Release drag if gesture changed away from okay
        if action != 'hold_drag' and self.drag_active:
            pyautogui.mouseUp(button='left')
            self.drag_active = False
            self.drag_x      = None
            self.drag_y      = None

        if action == 'left_click':
            if self._can_act('lclick'):
                self.mouse.left_click()
                return "LEFT CLICK"

        elif action == 'right_click':
            if self._can_act('rclick'):
                self.mouse.right_click()
                return "RIGHT CLICK"

        elif action == 'double_click':
            if self._can_act('dclick'):
                self.mouse.double_click()
                return "DOUBLE CLICK"

        elif action == 'scroll':
            cur_y = lms[0].y
            if self.prev_scroll_y is not None:
                delta = cur_y - self.prev_scroll_y
                if abs(delta) > SCROLL_THRESHOLD:
                    if delta > 0:
                        self.mouse.scroll('down', SCROLL_AMOUNT)
                        self.prev_scroll_y = cur_y
                        return "Scrolling DOWN"
                    else:
                        self.mouse.scroll('up', SCROLL_AMOUNT)
                        self.prev_scroll_y = cur_y
                        return "Scrolling UP"
            self.prev_scroll_y = cur_y
            return ""

        elif action == 'hold_drag':
            cur_x = lms[9].x   # middle finger MCP — stable drag anchor
            cur_y = lms[9].y
            if not self.drag_active:
                # First frame of okay — press and hold
                pyautogui.mouseDown(button='left')
                self.drag_active = True
                self.drag_x      = cur_x
                self.drag_y      = cur_y
                return "DRAG HOLD"
            else:
                # Continuing — move cursor while button held
                if self.drag_x is not None:
                    dx = (cur_x - self.drag_x) * MOVE_SENSITIVITY
                    dy = (cur_y - self.drag_y) * MOVE_SENSITIVITY
                    if abs(dx) > 0.001 or abs(dy) > 0.001:
                        cx, cy = self.mouse.get_cursor_position()
                        new_x = max(0, min(self.mouse.screen_width  - 1,
                                           cx + int(dx * self.mouse.screen_width)))
                        new_y = max(0, min(self.mouse.screen_height - 1,
                                           cy + int(dy * self.mouse.screen_height)))
                        pyautogui.moveTo(new_x, new_y)
                        self.drag_x = cur_x
                        self.drag_y = cur_y
                        return "DRAGGING"
                self.drag_x = cur_x
                self.drag_y = cur_y
                return "DRAG HOLD"

        return ""

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return

        show_mini = False   # Mini window disabled
        paused    = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            gesture     = None
            action_text = ""

            if not paused:
                # ── MediaPipe: detect landmarks ───────────────────────────
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.mp_detector.detect(mp_img)

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]

                    # ── TensorFlow: classify every 2nd frame (cached otherwise)
                    self.tf_frame_skip += 1
                    if self.tf_frame_skip % 2 == 0:
                        raw_g, conf = self._predict_gesture(lms)
                        self.last_gesture = raw_g
                    else:
                        raw_g = self.last_gesture   # reuse last result
                    gesture = self._stable_gesture(raw_g)

                    # Draw skeleton in gesture colour
                    skel_color = GESTURE_COLORS.get(gesture, (0, 220, 0))
                    self._draw_skeleton(frame, lms, skel_color)

                    # Draw bounding box
                    x1, y1, x2, y2 = self._get_bbox(frame, lms)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), skel_color, 2)

                    # ── Execute action ────────────────────────────────────
                    if gesture and gesture != 'unknown':
                        action_text = self._execute(gesture, lms)

                    # Reset scroll state when not in fist gesture
                    if gesture != 'fist':
                        self.prev_scroll_y = None

                else:
                    # No hand — reset all state
                    self.history.clear()
                    self.prev_scroll_y = None
                    self.prev_palm_x   = None
                    self.prev_palm_y   = None
                    if self.drag_active:
                        pyautogui.mouseUp(button='left')
                        self.drag_active = False
                    self.drag_x = None
                    self.drag_y = None

            # ── Render main window ────────────────────────────────────────
            fps_val = self._fps()
            self._draw_main_ui(frame, gesture, action_text, fps_val)
            cv2.imshow("Hand Gesture Controller", frame)

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('m'):
                show_mini = not show_mini
                print(f"Mini window {'ON' if show_mini else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()
        self.mp_detector.close()
        print("Goodbye!")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        GestureController().run()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()