"""
Data Collection Script - Landmark Based
Saves MediaPipe landmark coordinates as CSV rows (not images).
63 features per sample: 21 landmarks x (x, y, z) normalized to wrist.

Run:  python collect_data.py
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, csv, urllib.request

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import (GESTURES, SAMPLES_PER_GESTURE,
                        MEDIAPIPE_MODEL_PATH, MEDIAPIPE_MODEL_URL)
except ImportError:
    GESTURES             = {0:'palm',1:'index',2:'peace',3:'fist',4:'thumb_pinky',5:'okay'}
    SAMPLES_PER_GESTURE  = 500
    MEDIAPIPE_MODEL_PATH = 'hand_landmarker.task'
    MEDIAPIPE_MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
                            'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')

DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'landmarks.csv')

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(5,17)
]

GESTURE_COLORS = {
    'palm':(200,200,200),'index':(255,255,0),'peace':(0,255,0),
    'fist':(0,0,255),'thumb_pinky':(0,165,255),'okay':(255,0,255)
}

# ── Download MediaPipe model if needed ────────────────────────────────────────
if not os.path.exists(MEDIAPIPE_MODEL_PATH):
    print("Downloading hand landmarker model (~13 MB)...")
    urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
    print("  Model downloaded")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_landmarks(lms):
    """
    Normalize 21 landmarks relative to wrist (landmark 0).
    Returns a flat list of 63 values in [-1, 1].
    Makes the model invariant to hand position and scale.
    """
    base_x, base_y, base_z = lms[0].x, lms[0].y, lms[0].z
    coords = []
    for lm in lms:
        coords.append(lm.x - base_x)
        coords.append(lm.y - base_y)
        coords.append(lm.z - base_z)
    max_val = max(abs(v) for v in coords) or 1.0
    return [v / max_val for v in coords]


def draw_skeleton(frame, lms, color=(0, 220, 0)):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
    cv2.circle(frame, pts[0], 8, (0, 0, 255), -1)   # wrist in red


# ── Collection loop ───────────────────────────────────────────────────────────

def collect_gesture(gesture_id, gesture_name, num_samples, csv_writer):
    """Collect landmark samples for one gesture. Returns count saved."""
    print(f"\n{'='*60}")
    print(f"  Gesture {gesture_id+1}/6 : {gesture_name.upper()}")
    print(f"{'='*60}")
    print(f"  Target  : {num_samples} samples")
    print(f"  SPACE   : start / pause capturing")
    print(f"  Q       : done with this gesture\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    capturing = False
    count     = 0
    color     = GESTURE_COLORS.get(gesture_name, (0, 255, 0))

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # --- MediaPipe detection ---
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        hand_detected = bool(result.hand_landmarks)

        if hand_detected:
            lms = result.hand_landmarks[0]
            draw_skeleton(frame, lms, color)

            if capturing:
                features = normalize_landmarks(lms)
                csv_writer.writerow([gesture_id] + features)
                count += 1

        # --- UI overlay ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        status_text  = "CAPTURING" if (capturing and hand_detected) else (
                       "PAUSED"    if capturing else "Press SPACE to start")
        status_color = (0, 255, 0) if (capturing and hand_detected) else (
                       (0, 165, 255) if capturing else (0, 255, 255))

        cv2.putText(frame, f"Gesture: {gesture_name.upper()}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(frame, f"Status: {status_text}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Sample counter + progress bar
        progress = count / num_samples
        bar_w    = int(w * progress)
        cv2.rectangle(frame, (0, h - 8), (bar_w, h), color, -1)
        cv2.putText(frame, f"{count} / {num_samples}",
                    (w - 160, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        if not hand_detected:
            cv2.putText(frame, "No hand detected",
                        (w//2 - 110, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            capturing = not capturing
            print(f"  {'Started' if capturing else 'Paused'}  [{count}/{num_samples}]")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"  Saved {count} samples for '{gesture_name}'")
    return count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  LANDMARK DATA COLLECTION")
    print("="*60)
    print(f"  Gestures         : {', '.join(GESTURES.values())}")
    print(f"  Samples/gesture  : {SAMPLES_PER_GESTURE}")
    print(f"  Output           : {CSV_PATH}")
    print(f"  Features/sample  : 63  (21 landmarks x x,y,z)")
    print("="*60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Determine if appending or creating fresh
    file_exists = os.path.exists(CSV_PATH)
    mode        = 'a' if file_exists else 'w'
    if file_exists:
        print("\n  Existing data found - appending new samples.")
    else:
        print("\n  Creating new dataset.")

    total = 0
    with open(CSV_PATH, mode, newline='') as f:
        writer = csv.writer(f)
        # Header row only when creating fresh
        if not file_exists:
            header = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ('x','y','z')]
            writer.writerow(header)

        for gesture_id, gesture_name in GESTURES.items():
            input(f"\n  Press ENTER when ready to collect '{gesture_name}'...")
            count  = collect_gesture(gesture_id, gesture_name,
                                     SAMPLES_PER_GESTURE, writer)
            total += count

    print("\n" + "="*60)
    print("  COLLECTION COMPLETE!")
    print("="*60)
    print(f"  Total samples saved : {total}")
    print(f"  CSV file            : {CSV_PATH}")
    print("\n  Next step: open train_model.ipynb and run all cells.")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
