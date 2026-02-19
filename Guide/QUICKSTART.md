# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (5 minutes)

```bash
# Navigate to project folder
cd hand-gesture-controller

# Install required packages
pip install -r requirements.txt
```

### Step 2: Train Your Model (30 minutes)

```bash
# Start Jupyter Notebook
jupyter notebook

# Open: notebooks/train_model.ipynb
# Follow the instructions in the notebook to:
# 1. Collect gesture data (500 samples per gesture)
# 2. Train the model (takes 10-30 minutes)
# 3. Save the trained model
```

**Data Collection Tips**:
- 📸 Ensure good lighting
- 🖐️ Make clear, distinct gestures
- 🔄 Add variation (angles, distances)
- ⏸️ Press SPACE to start/stop capturing
- ➡️ Press ENTER to move to next gesture

### Step 3: Run the Application (Instant!)

```bash
# Navigate to src folder
cd src

# Run the gesture controller
python gesture_controller.py
```

**Controls**:
- `q` - Quit
- `s` - Toggle mini window
- `p` - Pause/Resume

---

## 📋 Gesture Cheat Sheet

Print this and keep it handy while using the application!

```
┌─────────────────────────────────────────────────────┐
│           HAND GESTURE CONTROLLER                    │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ✋ PALM           →  Move cursor                   │
│  👆 INDEX          →  Left click                    │
│  ✌️  PEACE         →  Right click                   │
│  👊 FIST           →  Scroll (move hand up/down)   │
│  🤙 THUMB+PINKY    →  Open App 1 (Notepad)         │
│  👌 OKAY           →  Open App 2 (Calculator)       │
│                                                      │
├─────────────────────────────────────────────────────┤
│  KEYBOARD SHORTCUTS:                                 │
│  • Q - Quit application                             │
│  • S - Toggle mini window                           │
│  • P - Pause/Resume recognition                     │
└─────────────────────────────────────────────────────┘
```

---

## 🎯 Performance Tips

### For Best Results:
- ✅ Use in well-lit environment
- ✅ Keep hand clearly visible
- ✅ Make distinct gestures
- ✅ Position camera at eye level
- ✅ Clean background preferred

### If Having Issues:
- ❌ Low accuracy? Collect more training data
- ❌ Slow performance? Close other apps
- ❌ Gesture not detected? Check lighting
- ❌ Jittery cursor? Increase smoothing value

---

## 🔧 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Camera not opening | Check camera permissions, close other apps using camera |
| Model not found | Complete training in Jupyter notebook first |
| Low FPS | Reduce resolution or close background apps |
| Gestures confused | Make gestures more distinct, retrain with more data |
| Mouse too fast/slow | Adjust smoothing parameter in config |

---

## 📊 Training Checklist

Use this checklist to ensure good training:

- [ ] Python environment activated
- [ ] All dependencies installed
- [ ] Camera working properly
- [ ] Good lighting setup
- [ ] Collected data for "palm" (500 samples)
- [ ] Collected data for "index" (500 samples)
- [ ] Collected data for "peace" (500 samples)
- [ ] Collected data for "fist" (500 samples)
- [ ] Collected data for "thumb_pinky" (500 samples)
- [ ] Collected data for "okay" (500 samples)
- [ ] Model training completed
- [ ] Model saved to models/ folder
- [ ] Training accuracy > 90%
- [ ] Tested model with live camera

---

## 🎓 Next Steps After Setup

1. **Test All Gestures**: Make sure each gesture is recognized correctly
2. **Adjust Sensitivity**: Modify confidence threshold if needed
3. **Customize Apps**: Change which apps open with gestures
4. **Add More Gestures**: Collect data for new gestures and retrain
5. **Optimize**: Adjust parameters for your hardware

---

## 💡 Pro Tips

### For Better Accuracy:
- Collect data from multiple angles
- Include both hands (if using multi-hand)
- Train for 50+ epochs
- Use data augmentation (already included)

### For Better Performance:
- Use dedicated GPU if available
- Close unnecessary browser tabs
- Disable other camera applications
- Use lower resolution for slower computers

### For Better Experience:
- Position camera 1-2 feet from your face
- Sit in consistent lighting
- Use plain background
- Keep hands steady for 1 second per gesture

---

## 🎮 Usage Examples

### Example 1: Browse the Web
1. Use PALM to move cursor to address bar
2. Use INDEX to click
3. Type your URL
4. Use FIST to scroll down pages

### Example 2: Work with Documents
1. THUMB+PINKY to open Notepad
2. Use PEACE to right-click for menu
3. FIST to scroll through document
4. INDEX to click buttons

### Example 3: Quick Calculations
1. OKAY gesture to open Calculator
2. Use PALM + INDEX to enter numbers
3. INDEX to click equals button

---

## 📱 Stay Updated

Check the main README.md for:
- Detailed troubleshooting
- Advanced configuration
- API documentation
- Contributing guidelines

---

**Ready to control your computer with gestures? Let's go! 🚀**
