"""
Hand Gesture Controller
Pipeline: MediaPipe landmarks -> TensorFlow model -> Mouse/System actions
Gestures: palm=move, index=left_click, peace=right_click, fist=scroll,
          thumb_pinky=double_click, okay=none, pinch=click+drag

UI: PyQt5 (main window + always-on-top mini overlay)
"""

import sys
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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

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
    GESTURES                   = {0:'palm',1:'index',2:'peace',3:'fist',4:'thumb_pinky',5:'okay'}
    NUM_CLASSES                = 6
    INPUT_FEATURES             = 63
    MODEL_PATH                 = 'models/gesture_model.h5'
    GESTURE_MAPPING_PATH       = 'models/gesture_mapping.json'
    CONFIDENCE_THRESHOLD       = 0.85
    MEDIAPIPE_MODEL_PATH       = 'hand_landmarker.task'
    MEDIAPIPE_MODEL_URL        = ('https://storage.googleapis.com/mediapipe-models/'
                                  'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
    DETECTION_CONFIDENCE       = 0.7
    TRACKING_CONFIDENCE        = 0.5
    CAMERA_INDEX               = 0
    CAMERA_WIDTH               = 640
    CAMERA_HEIGHT              = 480
    MOUSE_SMOOTHING            = 5
    MOVE_SENSITIVITY           = 0.1
    ACTION_COOLDOWN            = 0.8
    SCROLL_AMOUNT              = 30
    SCROLL_THRESHOLD           = 0.008
    GESTURE_HISTORY_LENGTH     = 6
    GESTURE_STABILITY_REQUIRED = 4
    SHOW_MINI_WINDOW           = True
    MINI_WIDTH                 = 280
    MINI_HEIGHT                = 210
    MINI_BORDER_COLOR          = (0, 255, 0)
    MINI_BORDER_THICKNESS      = 2
    GESTURE_ACTIONS            = {'palm':'move_cursor','index':'left_click',
                                   'peace':'right_click','fist':'scroll',
                                   'thumb_pinky':'double_click','okay':'hold_drag'}
    GESTURE_COLORS             = {'palm':(200,200,200),'index':(255,255,0),
                                   'peace':(0,255,0),'fist':(0,0,255),
                                   'thumb_pinky':(0,165,255),'okay':(0,255,255),
                                   'unknown':(160,160,160)}
    SHOW_FPS                   = True
    SHOW_CONFIDENCE            = True


# ── Skeleton connections ──────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(5,17)
]

PINCH_THRESHOLD = 0.07


# ── BGR colour → CSS hex helper ───────────────────────────────────────────────
def bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


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


# ══════════════════════════════════════════════════════════════════════════════
# Worker thread — all camera / ML / mouse logic runs here, never on GUI thread
# ══════════════════════════════════════════════════════════════════════════════
class GestureWorker(QThread):
    # Emitted every frame with (annotated_frame_bgr, mini_frame_bgr, gesture, action, fps, confidence)
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, str, str, float, float)
    # Emitted when thread finishes
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._paused    = False
        self._running   = True
        self._ctrl      = None     # GestureController; built inside run()

    # ── public control ────────────────────────────────────────────────────────
    def toggle_pause(self):
        self._paused = not self._paused
        print("PAUSED" if self._paused else "RESUMED")

    def stop(self):
        self._running = False

    # ── thread body ───────────────────────────────────────────────────────────
    def run(self):
        try:
            self._ctrl = GestureController()
        except Exception as e:
            import traceback
            print(f"Init error: {e}")
            traceback.print_exc()
            self.finished.emit()
            return

        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            self.finished.emit()
            return

        ctrl = self._ctrl

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            frame       = cv2.flip(frame, 1)
            gesture     = None
            action_text = ""

            if not self._paused:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = ctrl.mp_detector.detect(mp_img)

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]

                    ctrl.tf_frame_skip += 1
                    if ctrl.tf_frame_skip % 2 == 0:
                        raw_g, _ = ctrl._predict_gesture(lms)
                        ctrl.last_gesture = raw_g
                    else:
                        raw_g = ctrl.last_gesture
                    gesture = ctrl._stable_gesture(raw_g)

                    skel_color = GESTURE_COLORS.get(gesture, (0, 220, 0))
                    ctrl._draw_skeleton(frame, lms, skel_color)

                    x1, y1, x2, y2 = ctrl._get_bbox(frame, lms)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), skel_color, 2)

                    if gesture and gesture != 'unknown':
                        action_text = ctrl._execute(gesture, lms)

                    if gesture != 'fist':
                        ctrl.prev_scroll_y = None
                else:
                    ctrl.history.clear()
                    ctrl.prev_scroll_y = None
                    ctrl.prev_palm_x   = None
                    ctrl.prev_palm_y   = None
                    if ctrl.drag_active:
                        pyautogui.mouseUp(button='left')
                        ctrl.drag_active = False
                    ctrl.drag_x = None
                    ctrl.drag_y = None

            fps_val  = ctrl._fps()
            mini     = ctrl._build_mini_frame(frame, gesture)
            annotated = frame.copy()
            ctrl._draw_main_ui(annotated, gesture, action_text, fps_val)

            self.frame_ready.emit(
                annotated, mini,
                gesture or 'unknown',
                action_text,
                fps_val,
                ctrl.confidence
            )

        cap.release()
        if ctrl.mp_detector:
            ctrl.mp_detector.close()
        print("Goodbye!")
        self.finished.emit()


# ══════════════════════════════════════════════════════════════════════════════
# Mini overlay window — always on top, frameless
# ══════════════════════════════════════════════════════════════════════════════
class MiniOverlay(QWidget):

    def __init__(self):
        super().__init__()
        # Frameless + always on top + tool (no taskbar entry)
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedSize(MINI_WIDTH, MINI_HEIGHT + 24)   # extra 24 px for title bar

        # ── tiny drag-to-move title bar ───────────────────────────────────
        title_bar = QWidget(self)
        title_bar.setFixedHeight(24)
        title_bar.setStyleSheet("background:#1a1a2e;")
        title_bar.setCursor(Qt.SizeAllCursor)

        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(6, 0, 4, 0)
        tb_layout.setSpacing(0)

        tb_lbl = QLabel("● Gesture Mini View", title_bar)
        tb_lbl.setStyleSheet("color:#00ff88; font-size:10px; font-weight:600;")
        tb_layout.addWidget(tb_lbl)
        tb_layout.addStretch()

        # Store for drag
        self._drag_pos = None
        title_bar.mousePressEvent   = self._tb_press
        title_bar.mouseMoveEvent    = self._tb_move
        title_bar.mouseReleaseEvent = self._tb_release

        # ── camera feed ───────────────────────────────────────────────────
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(MINI_WIDTH, MINI_HEIGHT)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#000;")

        # ── gesture label strip ───────────────────────────────────────────
        self.gesture_label = QLabel("–", self)
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.gesture_label.setStyleSheet(
            "color:#00ff88; background:#0d0d1a; font-size:11px; font-weight:700; padding:2px;"
        )
        self.gesture_label.setFixedWidth(MINI_WIDTH)

        # ── layout ────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(title_bar)
        root.addWidget(self.video_label)

        # Position: top-right of screen
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - MINI_WIDTH - 10, 10)

        # Raise to ensure it is truly on top
        self.raise_()
        self.activateWindow()

    # ── drag helpers ──────────────────────────────────────────────────────────
    def _tb_press(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_pos = ev.globalPos() - self.frameGeometry().topLeft()
    def _tb_move(self, ev):
        if self._drag_pos and ev.buttons() == Qt.LeftButton:
            self.move(ev.globalPos() - self._drag_pos)
    def _tb_release(self, _ev):
        self._drag_pos = None

    # ── update frame ──────────────────────────────────────────────────────────
    def update_frame(self, bgr_frame, gesture):
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

        # Coloured border matching gesture
        color_bgr = GESTURE_COLORS.get(gesture, (80, 80, 80))
        hex_col   = bgr_to_hex(color_bgr)
        self.setStyleSheet(f"border: {MINI_BORDER_THICKNESS}px solid {hex_col};")

        # Gesture text
        self.gesture_label.setText(gesture.upper() if gesture else "–")
        self.gesture_label.setStyleSheet(
            f"color:{hex_col}; background:#0d0d1a; "
            "font-size:11px; font-weight:700; padding:2px;"
        )

        # Keep on top every frame (combats other apps stealing focus)
        self.raise_()


# ══════════════════════════════════════════════════════════════════════════════
# Main window
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Controller")
        self.setMinimumSize(860, 600)
        self._apply_dark_theme()

        # ── central widget ────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── header bar ────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(50)
        header.setStyleSheet("background:#0d0d1a; border-bottom:1px solid #1e1e3a;")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(14, 0, 14, 0)

        title_lbl = QLabel("✋  HAND GESTURE CONTROLLER")
        title_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title_lbl.setStyleSheet("color:#00ffcc; letter-spacing:2px;")
        h_lay.addWidget(title_lbl)
        h_lay.addStretch()

        self.fps_lbl = QLabel("FPS: –")
        self.fps_lbl.setStyleSheet("color:#00ff88; font-size:13px; font-weight:600;")
        h_lay.addWidget(self.fps_lbl)

        root_layout.addWidget(header)

        # ── body: video + sidebar ─────────────────────────────────────────
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(8, 8, 8, 8)
        body_lay.setSpacing(8)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#000; border-radius:6px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body_lay.addWidget(self.video_label, stretch=3)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(210)
        sidebar.setStyleSheet(
            "background:#0d0d1a; border-radius:6px; border:1px solid #1e1e3a;"
        )
        side_lay = QVBoxLayout(sidebar)
        side_lay.setContentsMargins(10, 14, 10, 14)
        side_lay.setSpacing(10)

        s_title = QLabel("STATUS")
        s_title.setStyleSheet("color:#555; font-size:10px; font-weight:700; letter-spacing:2px;")
        side_lay.addWidget(s_title)

        self.gesture_badge = QLabel("–")
        self.gesture_badge.setAlignment(Qt.AlignCenter)
        self.gesture_badge.setFixedHeight(40)
        self.gesture_badge.setStyleSheet(
            "background:#1a1a2e; color:#fff; border-radius:6px;"
            "font-size:16px; font-weight:700; border:1px solid #333;"
        )
        side_lay.addWidget(self.gesture_badge)

        self.action_lbl = QLabel("–")
        self.action_lbl.setAlignment(Qt.AlignCenter)
        self.action_lbl.setWordWrap(True)
        self.action_lbl.setStyleSheet(
            "color:#00ffcc; font-size:12px; font-weight:600; padding:4px;"
        )
        side_lay.addWidget(self.action_lbl)

        # Confidence bar row
        conf_title = QLabel("CONFIDENCE")
        conf_title.setStyleSheet("color:#555; font-size:10px; font-weight:700; letter-spacing:2px;")
        side_lay.addWidget(conf_title)

        self.conf_bar_bg = QLabel()
        self.conf_bar_bg.setFixedHeight(16)
        self.conf_bar_bg.setStyleSheet("background:#1a1a2e; border-radius:4px;")
        side_lay.addWidget(self.conf_bar_bg)

        self.conf_lbl = QLabel("0%")
        self.conf_lbl.setAlignment(Qt.AlignCenter)
        self.conf_lbl.setStyleSheet("color:#888; font-size:11px;")
        side_lay.addWidget(self.conf_lbl)

        # Gesture reference
        ref_title = QLabel("GESTURES")
        ref_title.setStyleSheet(
            "color:#555; font-size:10px; font-weight:700; letter-spacing:2px; margin-top:6px;"
        )
        side_lay.addWidget(ref_title)

        for g, a in GESTURE_ACTIONS.items():
            row = QLabel(f"<b>{g}</b>  →  {a.replace('_',' ')}")
            row.setStyleSheet("color:#aaa; font-size:10px;")
            side_lay.addWidget(row)

        side_lay.addStretch()

        # Control buttons
        btn_style = (
            "QPushButton{background:#1e1e3a; color:#00ffcc; border-radius:5px;"
            "font-weight:700; padding:7px; border:1px solid #2a2a4a;}"
            "QPushButton:hover{background:#2a2a5a;}"
            "QPushButton:pressed{background:#111130;}"
        )

        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setStyleSheet(btn_style)
        self.pause_btn.clicked.connect(self._toggle_pause)
        side_lay.addWidget(self.pause_btn)

        self.mini_btn = QPushButton("🪟  Mini Window ON")
        self.mini_btn.setStyleSheet(btn_style)
        self.mini_btn.clicked.connect(self._toggle_mini)
        side_lay.addWidget(self.mini_btn)

        body_lay.addWidget(sidebar, stretch=0)
        root_layout.addWidget(body, stretch=1)

        # ── status bar hint ───────────────────────────────────────────────
        self.statusBar().setStyleSheet(
            "background:#0d0d1a; color:#555; font-size:10px;"
        )
        self.statusBar().showMessage(
            "  Pause = P key or button    |    Mini overlay: button or M key    |    Q / close = quit"
        )

        # ── mini window ───────────────────────────────────────────────────
        self.mini_window = MiniOverlay()
        self.mini_window.show() if SHOW_MINI_WINDOW else self.mini_window.hide()
        self._mini_visible = SHOW_MINI_WINDOW
        if not self._mini_visible:
            self.mini_btn.setText("🪟  Mini Window OFF")

        # ── worker thread ─────────────────────────────────────────────────
        self.worker = GestureWorker()
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.finished.connect(self.close)
        self.worker.start()

        # ── confidence bar width tracking ─────────────────────────────────
        self._conf_bar_width = 0

    # ── dark theme ────────────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #080812; }
            QWidget      { background: #080812; color: #ddd; }
            QStatusBar   { background: #0d0d1a; color: #555; }
        """)

    # ── frame update slot ─────────────────────────────────────────────────────
    def _on_frame(self, annotated, mini_bgr, gesture, action, fps, confidence):
        # Main video
        rgb  = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

        # FPS
        if SHOW_FPS:
            self.fps_lbl.setText(f"FPS: {int(fps)}")

        # Gesture badge
        color_bgr = GESTURE_COLORS.get(gesture, (80, 80, 80))
        hex_col   = bgr_to_hex(color_bgr)
        self.gesture_badge.setText(gesture.upper() if gesture and gesture != 'unknown' else "–")
        self.gesture_badge.setStyleSheet(
            f"background:{hex_col}22; color:{hex_col}; border-radius:6px;"
            f"font-size:16px; font-weight:700; border:2px solid {hex_col};"
        )

        # Action
        self.action_lbl.setText(action if action else "–")

        # Confidence bar (CSS width via fixed-width trick using padding)
        if SHOW_CONFIDENCE:
            bar_w  = self.conf_bar_bg.width() or 190
            filled = int(bar_w * confidence)
            pct    = int(confidence * 100)
            self.conf_lbl.setText(f"{pct}%")
            self.conf_bar_bg.setStyleSheet(
                f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                f"stop:0 {hex_col}, stop:{confidence:.2f} {hex_col},"
                f"stop:{min(confidence+0.001,1):.3f} #1a1a2e, stop:1 #1a1a2e);"
                "border-radius:4px;"
            )

        # Mini window
        if self._mini_visible:
            self.mini_window.update_frame(mini_bgr, gesture)

    # ── button handlers ───────────────────────────────────────────────────────
    def _toggle_pause(self):
        self.worker.toggle_pause()
        paused = self.pause_btn.text().startswith("▶")
        self.pause_btn.setText("⏸  Pause" if paused else "▶  Resume")

    def _toggle_mini(self):
        self._mini_visible = not self._mini_visible
        if self._mini_visible:
            self.mini_window.show()
            self.mini_window.raise_()
            self.mini_btn.setText("🪟  Mini Window ON")
        else:
            self.mini_window.hide()
            self.mini_btn.setText("🪟  Mini Window OFF")

    # ── keyboard shortcuts ────────────────────────────────────────────────────
    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_Q:
            self.close()
        elif key == Qt.Key_P:
            self._toggle_pause()
        elif key == Qt.Key_M:
            self._toggle_mini()

    # ── clean shutdown ────────────────────────────────────────────────────────
    def closeEvent(self, ev):
        self.worker.stop()
        self.worker.wait(3000)
        self.mini_window.hide()
        ev.accept()


# ══════════════════════════════════════════════════════════════════════════════
# GestureController (logic only — no GUI, no cv2.imshow)
# ══════════════════════════════════════════════════════════════════════════════
class GestureController:

    def __init__(self):
        self._load_tf_model()
        self._init_mediapipe()
        self.mouse           = MouseController(smoothing=MOUSE_SMOOTHING)
        self.history         = deque(maxlen=GESTURE_HISTORY_LENGTH)
        self.last_action     = {}
        self.prev_scroll_y   = None
        self.prev_palm_x     = None
        self.prev_palm_y     = None
        self.fps_q           = deque(maxlen=10)
        self.prev_time       = time.time()
        self.confidence      = 0.0
        self.last_gesture    = 'unknown'
        self.tf_frame_skip   = 0
        self.drag_active     = False
        self.drag_x          = None
        self.drag_y          = None
        dummy = np.zeros((1, INPUT_FEATURES), dtype=np.float32)
        self.tf_model(dummy, training=False)
        print("  TF model warmed up")
        self._print_startup()

    # ── init helpers ──────────────────────────────────────────────────────────

    def _load_tf_model(self):
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
        print("  HAND GESTURE CONTROLLER  (TF + MediaPipe + PyQt5 UI)")
        print("="*58)
        for g, a in GESTURE_ACTIONS.items():
            print(f"  {g:14s} -> {a}")
        print(f"  {'pinch':14s} -> click + drag  (landmark-based)")
        print("-"*58)
        print("  Q = quit    P = pause / resume    M = mini window")
        print("="*58 + "\n")

    # ── per-frame helpers ─────────────────────────────────────────────────────

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
            if SHOW_CONFIDENCE:
                bar_len = int(200 * self.confidence)
                cv2.rectangle(frame, (tw + 34, 75), (tw + 34 + 200, 88), (50, 50, 50), -1)
                cv2.rectangle(frame, (tw + 34, 75), (tw + 34 + bar_len, 88), color, -1)
                cv2.putText(frame, f"{self.confidence*100:.0f}%",
                            (tw + 244, 88), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Action banner
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
        mini  = cv2.resize(frame, (MINI_WIDTH, MINI_HEIGHT))
        color = GESTURE_COLORS.get(gesture, MINI_BORDER_COLOR)
        cv2.rectangle(mini, (0, 0),
                      (MINI_WIDTH - 1, MINI_HEIGHT - 1),
                      color, MINI_BORDER_THICKNESS)
        if gesture and gesture != 'unknown':
            cv2.putText(mini, gesture.upper(),
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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        win = MainWindow()
        win.show()
        sys.exit(app.exec_())
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()