# ⚡ QUICK FIX - MediaPipe Error

## Got This Error?
```
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

## ✅ SOLUTION: Use Simple Method (2 Steps)

### Step 1: Collect Data
```bash
cd src
python simple_collect.py
```

**What this does:**
- Opens your camera with a GREEN BOX
- Press SPACE to start capturing
- Automatically goes through all 6 gestures
- No MediaPipe needed for collection!

**Instructions per gesture:**
1. Position your hand in the GREEN BOX
2. Press SPACE to start capturing
3. Move hand slightly for variation
4. Press SPACE to stop when you have enough samples
5. Press ENTER for next gesture

### Step 2: Train Model
```bash
python train_standalone.py
```

**What this does:**
- Loads all collected data
- Trains CNN model with TensorFlow
- Takes 10-30 minutes
- Creates gesture_model.h5 automatically

### Step 3: Run App
```bash
python gesture_controller.py
```

Done! 🎉

## 🎯 Complete Workflow

```bash
# 1. Navigate to project
cd hand-gesture-controller/src

# 2. Collect data (500 samples × 6 gestures)
python simple_collect.py

# 3. Train model (automatic)
python train_standalone.py

# 4. Run gesture controller
python gesture_controller.py
```

## 📋 Gestures to Collect

When collecting, make these gestures in the GREEN BOX:

1. **PALM** - Open hand, all fingers extended
2. **INDEX** - Only index finger pointing up
3. **PEACE** - Peace sign (index + middle)
4. **FIST** - Closed fist
5. **THUMB_PINKY** - Shaka/hang loose (thumb + pinky)
6. **OKAY** - OK sign (thumb + index circle)

## 💡 Tips for Better Data

- Good lighting is important
- Keep hand clearly in the GREEN BOX
- Add variation (slight angles, distances)
- Make distinct gestures
- Don't rush - quality over speed

## ❓ Still Having Issues?

### Issue: Camera not opening
**Fix:**
```bash
# Check camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

### Issue: Not enough samples
**Fix:** Collect at least 100 samples per gesture (500 recommended)

### Issue: Low accuracy after training
**Fix:** 
- Collect more data (1000 samples per gesture)
- Ensure good lighting during collection
- Make gestures more distinct
- Retrain with: `python train_standalone.py`

## 🔄 Alternative: Fix MediaPipe (Advanced)

If you prefer to use the Jupyter notebook:

```bash
# Uninstall current MediaPipe
pip uninstall mediapipe

# Install compatible version
pip install mediapipe==0.10.8

# Then use notebook normally
```

## ✨ Why Simple Method is Better

- ✅ No MediaPipe version conflicts
- ✅ Faster data collection
- ✅ Clearer visual feedback
- ✅ Works on all systems
- ✅ No Jupyter needed
- ✅ Automatic training

## 🎓 What You Get

After training, you'll have:
- `../models/gesture_model.h5` - Your trained model
- `../models/gesture_mapping.json` - Gesture labels
- `../models/training_history.png` - Training plots
- `../models/confusion_matrix.png` - Accuracy visualization

## 🚀 Next Steps

Once training is complete:
1. Check accuracy (should be >90%)
2. Run: `python gesture_controller.py`
3. Control your computer with gestures!

---

**Need More Help?**
- Check: `MEDIAPIPE_FIX.md` for detailed explanation
- Check: `README.md` for full documentation
- Check: `QUICKSTART.md` for quick reference

**Ready to start?**
```bash
cd src
python simple_collect.py
```

Let's go! 🖐️✨
