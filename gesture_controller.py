"""
Hand Gesture Controller
Pipeline: MediaPipe landmarks -> TensorFlow model -> Mouse/System actions

Normal mode gestures:
  palm        -> move cursor
  index       -> left click
  peace       -> right click
  fist        -> scroll
  thumb_pinky -> double click
  okay        -> hold & drag

Air Drawing mode (press D to toggle):
  index       -> draw
  fist        -> lift pen (idle)
  peace       -> eraser
  palm        -> clear canvas
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
    MOVE_SENSITIVITY          = 0.3
    ACTION_COOLDOWN           = 0.8
    SCROLL_AMOUNT             = 30
    SCROLL_THRESHOLD          = 0.008
    GESTURE_HISTORY_LENGTH    = 6
    GESTURE_STABILITY_REQUIRED= 4
    SHOW_MINI_WINDOW          = False
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


# ── Air Canvas ────────────────────────────────────────────────────────────────

DRAW_COLORS = [
    ('WHITE',  (255, 255, 255)),
    ('BLUE',   (255,  80,  20)),
    ('GREEN',  ( 20, 220,  20)),
    ('RED',    ( 20,  20, 255)),
    ('YELLOW', (  0, 220, 255)),
    ('PURPLE', (220,   0, 220)),
]
ERASER_COLOR  = (0, 0, 0)
ERASER_SIZE   = 40
DEFAULT_COLOR = 1          # BLUE
DEFAULT_THICK = 5

TB_HEIGHT   = 50
TB_PAD      = 8
COLOR_SZ    = 36
THICK_SLOTS = [3, 5, 8, 12]


class AirCanvas:
    """
    Transparent drawing canvas overlaid on the camera frame.
    Index fingertip (landmark 8) acts as the pen.
    """

    def __init__(self, width, height):
        self.w         = width
        self.h         = height
        self.active    = False
        self.canvas    = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_pt   = None
        self.color_idx = DEFAULT_COLOR
        self.thickness = DEFAULT_THICK
        self.eraser    = False
        self.state     = 'IDLE'
        self._build_toolbar()

    def _build_toolbar(self):
        self.color_rects = []
        x = TB_PAD
        y = TB_PAD
        for i, (name, bgr) in enumerate(DRAW_COLORS):
            self.color_rects.append((x, y, x + COLOR_SZ, y + COLOR_SZ, i))
            x += COLOR_SZ + TB_PAD

        x += TB_PAD
        self.thick_rects = []
        for t in THICK_SLOTS:
            self.thick_rects.append((x, y + COLOR_SZ//2 - 10, x + 32, y + COLOR_SZ//2 + 10, t))
            x += 32 + TB_PAD

        self.eraser_rect = (x, y, x + 60, y + COLOR_SZ)
        x += 70
        self.clear_rect  = (x, y, x + 50, y + COLOR_SZ)

    def _hit_toolbar(self, fx, fy):
        if fy > TB_HEIGHT:
            return None
        for (x1, y1, x2, y2, i) in self.color_rects:
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                return ('color', i)
        for (x1, y1, x2, y2, t) in self.thick_rects:
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                return ('thick', t)
        ex1, ey1, ex2, ey2 = self.eraser_rect
        if ex1 <= fx <= ex2 and ey1 <= fy <= ey2:
            return ('eraser',)
        cx1, cy1, cx2, cy2 = self.clear_rect
        if cx1 <= fx <= cx2 and cy1 <= fy <= cy2:
            return ('clear',)
        return None

    def toggle(self):
        self.active  = not self.active
        self.prev_pt = None
        self.state   = 'IDLE'

    def clear(self):
        self.canvas[:] = 0
        self.prev_pt   = None

    def update(self, gesture, finger_x, finger_y):
        """Update canvas state. Returns status string."""
        if not self.active:
            return ""

        fx = int(finger_x * self.w)
        fy = int(finger_y * self.h)

        # Toolbar hit
        tb = self._hit_toolbar(fx, fy)
        if tb:
            if tb[0] == 'color':
                self.color_idx = tb[1]
                self.eraser    = False
                return f"Color: {DRAW_COLORS[tb[1]][0]}"
            elif tb[0] == 'thick':
                self.thickness = tb[1]
                return f"Line: {tb[1]}"
            elif tb[0] == 'eraser':
                self.eraser = not self.eraser
                return "ERASER ON" if self.eraser else "ERASER OFF"
            elif tb[0] == 'clear':
                self.clear()
                return "Canvas cleared"

        if gesture == 'index':
            self.state  = 'ERASING' if self.eraser else 'DRAWING'
            color       = ERASER_COLOR if self.eraser else DRAW_COLORS[self.color_idx][1]
            if self.eraser:
                cv2.circle(self.canvas, (fx, fy), ERASER_SIZE, color, -1)
                self.prev_pt = None
            else:
                if self.prev_pt:
                    cv2.line(self.canvas, self.prev_pt, (fx, fy), color, self.thickness)
                else:
                    cv2.circle(self.canvas, (fx, fy), self.thickness // 2, color, -1)
                self.prev_pt = (fx, fy)

        elif gesture == 'fist':
            self.prev_pt = None
            self.state   = 'IDLE'

        elif gesture == 'peace':
            self.eraser  = True
            self.state   = 'ERASING'
            cv2.circle(self.canvas, (fx, fy), ERASER_SIZE, ERASER_COLOR, -1)
            self.prev_pt = None

        elif gesture == 'palm':
            self.clear()
            self.state = 'IDLE'
            return "Canvas cleared"

        if gesture not in ('peace', 'index') or not self.eraser:
            if gesture != 'peace':
                self.eraser = False

        return self.state

    def draw(self, frame, finger_x, finger_y, gesture):
        """Blend canvas onto frame and render toolbar + cursor."""
        if not self.active:
            return

        h, w = frame.shape[:2]
        fx = int(finger_x * w)
        fy = int(finger_y * h)

        # Blend drawing onto camera frame
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        cv2.add(bg, fg, frame)

        # Toolbar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, TB_HEIGHT + TB_PAD), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Colour swatches
        for (x1, y1, x2, y2, i) in self.color_rects:
            bgr = DRAW_COLORS[i][1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
            if i == self.color_idx and not self.eraser:
                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255,255,255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80,80,80), 1)

        # Thickness buttons
        for (x1, y1, x2, y2, t) in self.thick_rects:
            is_sel = (t == self.thickness)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70,70,70) if is_sel else (35,35,35), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120,120,120), 1)
            cy_m = (y1 + y2) // 2
            cv2.line(frame, (x1+4, cy_m), (x2-4, cy_m),
                     (255,255,255) if is_sel else (160,160,160), max(1, t//2))

        # Eraser button
        ex1, ey1, ex2, ey2 = self.eraser_rect
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0,200,200) if self.eraser else (50,50,50), -1)
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (120,120,120), 1)
        cv2.putText(frame, 'ERASE', (ex1+4, ey2-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)

        # Clear button
        cx1, cy1, cx2, cy2 = self.clear_rect
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (50,20,20), -1)
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (120,120,120), 1)
        cv2.putText(frame, 'CLR', (cx1+8, cy2-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,100,100), 1, cv2.LINE_AA)

        # State label top-right
        mode_col = {'IDLE':(180,180,180),'DRAWING':DRAW_COLORS[self.color_idx][1],
                    'ERASING':(0,200,200)}.get(self.state, (180,180,180))
        cv2.putText(frame, self.state, (w - 110, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_col, 2, cv2.LINE_AA)

        # Finger cursor
        if self.eraser or gesture == 'peace':
            cv2.circle(frame, (fx, fy), ERASER_SIZE, (0,220,220), 2)
        elif gesture == 'index':
            cv2.circle(frame, (fx, fy), self.thickness + 4, DRAW_COLORS[self.color_idx][1], -1)
            cv2.circle(frame, (fx, fy), self.thickness + 6, (255,255,255), 1)
        else:
            cv2.circle(frame, (fx, fy), 8, (100,100,100), 2)

        # Bottom hint
        cv2.putText(frame,
                    "D=exit draw | index=draw | fist=lift | peace=erase | palm=clear",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (140,140,140), 1, cv2.LINE_AA)


# ── Download MediaPipe model if needed ────────────────────────────────────────
if not os.path.exists(MEDIAPIPE_MODEL_PATH):
    print("Downloading hand landmarker model (~13 MB)...")
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
    print("  Model downloaded")


# ── Landmark normalisation ────────────────────────────────────────────────────
def normalize_landmarks(lms):
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
        self.mouse          = MouseController(smoothing=MOUSE_SMOOTHING)
        self.history        = deque(maxlen=GESTURE_HISTORY_LENGTH)
        self.last_action    = {}
        self.prev_scroll_y  = None
        self.prev_palm_x    = None
        self.prev_palm_y    = None
        self.fps_q          = deque(maxlen=10)
        self.prev_time      = time.time()
        self.confidence     = 0.0
        self.last_gesture   = 'unknown'
        self.tf_frame_skip  = 0
        self.drag_active    = False
        self.drag_x         = None
        self.drag_y         = None
        self.canvas         = None   # created in run() once we know frame size
        dummy = np.zeros((1, INPUT_FEATURES), dtype=np.float32)
        self.tf_model(dummy, training=False)
        print("  TF model warmed up")
        self._print_startup()

    # ── init ──────────────────────────────────────────────────────────────────

    def _load_tf_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "  1. python collect_data.py\n"
                "  2. Open train_model.ipynb and run all cells"
            )
        print(f"Loading TensorFlow model from {MODEL_PATH} ...")
        self.tf_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"  Model loaded  |  Input: {self.tf_model.input_shape}"
              f"  |  Output: {self.tf_model.output_shape}")
        if os.path.exists(GESTURE_MAPPING_PATH):
            with open(GESTURE_MAPPING_PATH) as f:
                raw = json.load(f)
            self.gesture_map = {int(k): v for k, v in raw.items()}
        else:
            self.gesture_map = GESTURES

    def _init_mediapipe(self):
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
        print("  NORMAL MODE:")
        for g, a in GESTURE_ACTIONS.items():
            print(f"    {g:14s} -> {a}")
        print("\n  AIR DRAWING MODE (press D):")
        print("    index          -> draw")
        print("    fist           -> lift pen")
        print("    peace          -> eraser")
        print("    palm           -> clear canvas")
        print("-"*58)
        print("  Q=quit   P=pause   D=toggle drawing")
        print("="*58 + "\n")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _predict_gesture(self, lms):
        features = normalize_landmarks(lms)
        probs    = self.tf_model(features, training=False).numpy()[0]
        top_id   = int(np.argmax(probs))
        top_conf = float(probs[top_id])
        self.confidence = top_conf
        if top_conf >= CONFIDENCE_THRESHOLD:
            return self.gesture_map.get(top_id, 'unknown'), top_conf
        return 'unknown', top_conf

    def _stable_gesture(self, g):
        self.history.append(g)
        if len(self.history) < self.history.maxlen:
            return None
        counts = {}
        for x in self.history:
            counts[x] = counts.get(x, 0) + 1
        best = max(counts, key=counts.get)
        return best if counts[best] >= GESTURE_STABILITY_REQUIRED else None

    def _can_act(self, key):
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

        if not (self.canvas and self.canvas.active):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 58), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, "HAND GESTURE CONTROLLER",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Q=quit  P=pause  D=draw",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (140, 140, 140), 1, cv2.LINE_AA)

            if gesture and gesture != 'unknown':
                color = GESTURE_COLORS.get(gesture, (255, 255, 255))
                label = gesture.upper()
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(frame, (8, 66), (tw + 24, 98), color, -1)
                cv2.putText(frame, label, (14, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
                if SHOW_CONFIDENCE:
                    bar_len = int(200 * self.confidence)
                    cv2.rectangle(frame, (tw+34, 75), (tw+234, 88), (50,50,50), -1)
                    cv2.rectangle(frame, (tw+34, 75), (tw+34+bar_len, 88), color, -1)
                    cv2.putText(frame, f"{self.confidence*100:.0f}%",
                                (tw+244, 88), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,255,255), 1, cv2.LINE_AA)

        if SHOW_FPS:
            cv2.putText(frame, f"FPS {int(fps_val)}",
                        (w - 100, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2, cv2.LINE_AA)

        if action_text:
            (tw, th), _ = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            tx = (w - tw) // 2
            ty = h - 50
            cv2.rectangle(frame, (tx-14, ty-th-10), (tx+tw+14, ty+10), (0,0,0), -1)
            cv2.putText(frame, action_text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)

    # ── action execution ──────────────────────────────────────────────────────

    def _execute(self, gesture, lms):
        action = GESTURE_ACTIONS.get(gesture, '')

        if action == 'move_cursor':
            cur_x = lms[9].x
            cur_y = lms[9].y
            if self.prev_palm_x is not None:
                dx = (cur_x - self.prev_palm_x) * MOVE_SENSITIVITY
                dy = (cur_y - self.prev_palm_y) * MOVE_SENSITIVITY
                if abs(dx) > 0.002 or abs(dy) > 0.002:
                    cx, cy = self.mouse.get_cursor_position()
                    new_x = max(0, min(self.mouse.screen_width  - 1,
                                       cx + int(dx * self.mouse.screen_width)))
                    new_y = max(0, min(self.mouse.screen_height - 1,
                                       cy + int(dy * self.mouse.screen_height)))
                    pyautogui.moveTo(new_x, new_y)
                    return "Moving Cursor"
            self.prev_palm_x = cur_x
            self.prev_palm_y = cur_y
            return ""

        self.prev_palm_x = None
        self.prev_palm_y = None

        if action != 'hold_drag' and self.drag_active:
            pyautogui.mouseUp(button='left')
            self.drag_active = False
            self.drag_x = None
            self.drag_y = None

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
            cur_x = lms[9].x
            cur_y = lms[9].y
            if not self.drag_active:
                pyautogui.mouseDown(button='left')
                self.drag_active = True
                self.drag_x = cur_x
                self.drag_y = cur_y
                return "DRAG HOLD"
            else:
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
        if not cap.isOpened():
            print(f"  Camera index {CAMERA_INDEX} failed, scanning 0-3...")
            cap = None
            for idx in range(4):
                test = cv2.VideoCapture(idx)
                if test.isOpened():
                    ret, _ = test.read()
                    if ret:
                        print(f"  Found camera at index {idx}")
                        cap = test
                        break
                test.release()
            if cap is None:
                print("\nERROR: No camera found.")
                print("  - Close any app using the camera (Zoom, Teams, OBS...)")
                print("  - Try changing CAMERA_INDEX in config.py")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read one frame to get actual dimensions
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from camera")
            cap.release()
            return
        actual_h, actual_w = frame.shape[:2]
        self.canvas = AirCanvas(actual_w, actual_h)

        paused  = False
        lms     = None   # keep last known landmarks for canvas draw when paused

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gesture     = None
            action_text = ""

            if not paused:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.mp_detector.detect(mp_img)

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]

                    # TF classify every 2nd frame
                    self.tf_frame_skip += 1
                    if self.tf_frame_skip % 2 == 0:
                        raw_g, _ = self._predict_gesture(lms)
                        self.last_gesture = raw_g
                    else:
                        raw_g = self.last_gesture
                    gesture = self._stable_gesture(raw_g)

                    # Skeleton + bounding box
                    skel_color = GESTURE_COLORS.get(gesture, (0, 220, 0))
                    self._draw_skeleton(frame, lms, skel_color)
                    x1, y1, x2, y2 = self._get_bbox(frame, lms)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), skel_color, 2)

                    fx, fy = lms[8].x, lms[8].y   # index fingertip

                    if self.canvas.active:
                        result_text = self.canvas.update(gesture, fx, fy)
                        if result_text:
                            action_text = result_text
                    else:
                        if gesture and gesture != 'unknown':
                            action_text = self._execute(gesture, lms)

                    if gesture != 'fist':
                        self.prev_scroll_y = None

                else:
                    lms = None
                    self.history.clear()
                    self.prev_scroll_y = None
                    self.prev_palm_x   = None
                    self.prev_palm_y   = None
                    if self.drag_active:
                        pyautogui.mouseUp(button='left')
                        self.drag_active = False
                    self.drag_x = None
                    self.drag_y = None
                    if self.canvas.active:
                        self.canvas.prev_pt = None

            # ── Render ───────────────────────────────────────────────────────
            fps_val = self._fps()

            if self.canvas.active:
                finger_x = lms[8].x if lms else 0.5
                finger_y = lms[8].y if lms else 0.5
                self.canvas.draw(frame, finger_x, finger_y, gesture)

            self._draw_main_ui(frame, gesture, action_text, fps_val)
            cv2.imshow("Hand Gesture Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('p'):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('d'):
                self.canvas.toggle()
                print(f"Air drawing {'ON' if self.canvas.active else 'OFF'}")

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