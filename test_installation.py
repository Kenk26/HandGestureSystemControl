"""
Installation Test Script
Verifies all dependencies and project structure are correct.
Run: python test_installation.py
"""

import sys
import os


def test_imports():
    print("Testing package imports...\n")
    packages = [
        ('cv2',        'OpenCV'),
        ('numpy',      'NumPy'),
        ('tensorflow', 'TensorFlow'),
        ('mediapipe',  'MediaPipe'),
        ('pyautogui',  'PyAutoGUI'),
        ('sklearn',    'scikit-learn'),
        ('pandas',     'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn',    'Seaborn'),
    ]
    all_ok = True
    for pkg, name in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', '')
            print(f"  OK   {name:20s} {ver}")
        except ImportError:
            print(f"  FAIL {name:20s} -- run: pip install {pkg}")
            all_ok = False
    return all_ok


def test_camera():
    print("\nTesting camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  FAIL  Camera not accessible")
            return False
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"  OK   Camera  ({w}x{h})")
            return True
        print("  FAIL  Cannot read frames")
        return False
    except Exception as e:
        print(f"  FAIL  {e}")
        return False


def test_gpu():
    print("\nTesting GPU / TensorFlow...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  OK   GPU available: {[g.name for g in gpus]}")
        else:
            print("  OK   No GPU (will use CPU - training still works)")
        print(f"  OK   TensorFlow {tf.__version__}")
        return True
    except Exception as e:
        print(f"  FAIL {e}")
        return False


def test_mediapipe():
    print("\nTesting MediaPipe...")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        print(f"  OK   MediaPipe {mp.__version__}  (Tasks API available)")
        return True
    except Exception as e:
        print(f"  FAIL {e}")
        return False


def test_pyautogui():
    print("\nTesting PyAutoGUI...")
    try:
        import pyautogui
        size = pyautogui.size()
        pos  = pyautogui.position()
        print(f"  OK   Screen {size[0]}x{size[1]}  |  Cursor at {pos}")
        return True
    except Exception as e:
        print(f"  FAIL {e}")
        return False


def test_structure():
    print("\nChecking project structure...")
    paths = [
        'data',
        'models',
        'config.py',
        'collect_data.py',
        'train_model.ipynb',
        'gesture_controller.py',
        'mouse_controller.py',
    ]
    all_ok = True
    for p in paths:
        exists = os.path.exists(p)
        status = "OK  " if exists else "MISS"
        print(f"  {status}  {p}")
        if not exists:
            all_ok = False
    return all_ok


def test_trained_model():
    print("\nChecking trained model...")
    model_path = 'models/gesture_model.h5'
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            m = tf.keras.models.load_model(model_path)
            print(f"  OK   Model loaded  |  Input {m.input_shape}  Output {m.output_shape}")
            return True
        except Exception as e:
            print(f"  FAIL  Model found but cannot load: {e}")
            return False
    else:
        print("  MISS  No trained model yet — run collect_data.py then train_model.ipynb")
        return None   # Not an error, just not done yet


def main():
    print("=" * 60)
    print("  HAND GESTURE CONTROLLER — INSTALLATION TEST")
    print("=" * 60)
    print(f"  Python {sys.version}\n")

    results = {
        "Package imports"  : test_imports(),
        "Camera"           : test_camera(),
        "TensorFlow / GPU" : test_gpu(),
        "MediaPipe"        : test_mediapipe(),
        "PyAutoGUI"        : test_pyautogui(),
        "Project structure": test_structure(),
        "Trained model"    : test_trained_model(),
    }

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passed = 0
    for name, result in results.items():
        if result is True:
            print(f"  PASS  {name}")
            passed += 1
        elif result is False:
            print(f"  FAIL  {name}")
        else:
            print(f"  SKIP  {name}  (not trained yet)")

    critical = sum(1 for k, v in results.items()
                   if v is False and k != "Trained model")

    print("\n" + "=" * 60)
    if critical == 0:
        print("  All critical checks passed!")
        if results["Trained model"] is not True:
            print("\n  Next steps:")
            print("    1. python collect_data.py")
            print("    2. Open train_model.ipynb and run all cells")
            print("    3. python gesture_controller.py")
        else:
            print("\n  Ready to run:  python gesture_controller.py")
    else:
        print(f"  {critical} critical issue(s) to fix before running.")
        print("  Install missing packages:  pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
