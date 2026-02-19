# Hand Gesture Controller

Control your computer mouse and applications using hand gestures! This project uses TensorFlow for gesture recognition and MediaPipe for real-time hand tracking.

## 🎯 Features

- **Real-time Hand Gesture Recognition** using TensorFlow CNN model
- **Mouse Control** - Move cursor with hand movements
- **Click Actions** - Left click, right click, and double click
- **Scroll Control** - Scroll up and down using gestures
- **Application Launcher** - Open applications with specific gestures
- **Live Camera Feed** - Mini overlay window showing your hand with gesture tracking
- **High Accuracy** - Trained CNN model with data augmentation
- **Smooth Performance** - Optimized for real-time performance

## 🖐️ Gesture Controls

| Gesture | Action | Description |
|---------|--------|-------------|
| ✋ **Open Palm** | Move Cursor | Move your hand to control the cursor |
| 👆 **Index Finger Up** | Left Click | Point with your index finger |
| ✌️ **Peace Sign** | Right Click | Show peace sign (index + middle finger) |
| 👊 **Fist** | Scroll | Close your hand and move up/down to scroll |
| 🤙 **Thumb + Pinky** | Open App 1 | Extend thumb and pinky (default: Notepad) |
| 👌 **Okay Sign** | Open App 2 | Make OK gesture (default: Calculator) |

## 📁 Project Structure

```
hand-gesture-controller/
├── data/
│   ├── raw/              # Raw gesture images (collected by you)
│   └── processed/        # Preprocessed data
├── models/
│   ├── gesture_model.h5  # Trained model (created after training)
│   └── gesture_mapping.json  # Gesture labels
├── notebooks/
│   └── train_model.ipynb # Jupyter notebook for training
├── src/
│   ├── gesture_controller.py  # Main application
│   ├── hand_detector.py       # Hand detection module
│   └── mouse_controller.py    # Mouse control logic
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Installation

### Step 1: Clone or Download the Project

Download and extract the project folder to your desired location.

### Step 2: Create Virtual Environment (Recommended)

```bash
# Navigate to project directory
cd hand-gesture-controller

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installation might take a few minutes as TensorFlow is a large package.

## 📊 Training the Model

### Step 1: Open Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to: notebooks/train_model.ipynb
```

### Step 2: Data Collection

The notebook will guide you through collecting gesture data:

1. **Run cells sequentially** in the notebook
2. **For each gesture**, you'll collect 500 samples:
   - Position your hand clearly in front of the camera
   - Press SPACE to start/stop capturing
   - Move your hand slightly while capturing to add variation
   - Good lighting is important!
   
3. **Gestures to collect**:
   - Palm (open hand)
   - Index (only index finger up)
   - Peace (index + middle finger)
   - Fist (closed hand)
   - Thumb + Pinky (shaka sign)
   - Okay (thumb + index forming circle)

**Tips for better data collection**:
- Use good lighting
- Try different hand positions and angles
- Include slight rotations
- Vary the distance from camera
- Keep background clean

### Step 3: Train the Model

After collecting data, continue running the notebook cells to:
- Preprocess the data
- Build the CNN model
- Train for 50 epochs (takes 10-30 minutes depending on your hardware)
- Evaluate model performance
- Save the trained model

**Expected Results**:
- Training accuracy: >95%
- Validation accuracy: >90%

## 🎮 Running the Application

### Step 1: Open VS Code

```bash
# Open project in VS Code
code .
```

### Step 2: Run the Main Application

```bash
# Navigate to src directory
cd src

# Run the gesture controller
python gesture_controller.py
```

### Step 3: Use the Application

Once running, you'll see:
- Main window with your camera feed
- Mini overlay window in the top-left corner showing hand tracking
- Gesture name and confidence displayed
- Current action shown at the bottom

**Keyboard Controls**:
- `q` - Quit the application
- `s` - Toggle mini window on/off
- `p` - Pause/Resume gesture recognition

## ⚙️ Configuration

### Customize Applications

Edit `src/gesture_controller.py` or use the configuration:

```python
# In mouse_controller.py, modify GestureActionMapper class:
self.app_mappings = {
    'open_app1': 'notepad',      # Change to your preferred app
    'open_app2': 'calculator'    # Change to your preferred app
}
```

**Application names by OS**:

| Application | Windows | macOS | Linux |
|------------|---------|-------|-------|
| Text Editor | `notepad` | `TextEdit` | `gedit` |
| Calculator | `calc` | `Calculator` | `gnome-calculator` |
| Browser | `chrome` | `Google Chrome` | `google-chrome` |
| File Manager | `explorer` | `Finder` | `nautilus` |

### Adjust Sensitivity

In `gesture_controller.py`:

```python
# Confidence threshold (0-1)
confidence_threshold = 0.7  # Higher = more strict

# Mouse smoothing (1-10)
smoothing = 7  # Higher = smoother but slower

# Action cooldown (seconds)
action_cooldown = 1.0  # Prevents repeated clicks
```

## 🔧 Troubleshooting

### Camera Not Working
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

### Model Not Found
- Make sure you've completed training in Jupyter notebook
- Check that `models/gesture_model.h5` exists
- Re-run the training notebook if needed

### Low Accuracy
- Collect more training data (1000+ samples per gesture)
- Ensure good lighting during data collection
- Train for more epochs
- Check that gestures are distinct from each other

### PyAutoGUI Permission Issues (macOS)
- Go to: System Preferences → Security & Privacy → Privacy → Accessibility
- Add Terminal or Python to allowed applications

### Performance Issues
- Close other applications
- Reduce camera resolution in `gesture_controller.py`:
  ```python
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  ```
- Use GPU if available (automatic with TensorFlow)

## 📈 Model Performance

The CNN model architecture:
- 3 Convolutional blocks with batch normalization
- Max pooling and dropout for regularization
- 2 Dense layers with dropout
- Softmax output for 6 gesture classes

**Training details**:
- Input size: 128x128 RGB images
- Data augmentation: rotation, shift, zoom, flip
- Optimizer: Adam
- Loss: Categorical crossentropy
- Callbacks: Early stopping, learning rate reduction

## 🎓 How It Works

### 1. Hand Detection
- MediaPipe detects 21 hand landmarks in real-time
- Extracts bounding box around the hand
- Normalizes hand region to 128x128 pixels

### 2. Gesture Recognition
- CNN model processes the hand image
- Outputs probability distribution over 6 gestures
- Uses gesture history for stability (reduces jitter)

### 3. Action Execution
- Maps recognized gesture to action
- Executes mouse/keyboard commands via PyAutoGUI
- Applies cooldown to prevent repeated actions

### 4. Visual Feedback
- Main window shows full camera feed with UI
- Mini window shows hand tracking overlay
- Real-time FPS, gesture name, and confidence display

## 🔮 Future Enhancements

Possible improvements:
- [ ] Add more gestures (zoom, drag, minimize, etc.)
- [ ] Custom gesture training interface
- [ ] Multi-hand support
- [ ] Gesture macros (sequences of actions)
- [ ] Voice command integration
- [ ] Web-based configuration panel
- [ ] Support for custom keyboard shortcuts

## 📝 Credits

**Technologies Used**:
- TensorFlow/Keras - Deep learning framework
- MediaPipe - Hand tracking
- OpenCV - Computer vision
- PyAutoGUI - Mouse/keyboard control
- NumPy - Numerical operations

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Contributions are welcome! Ideas:
- Improve model architecture
- Add new gestures
- Optimize performance
- Create better UI
- Add tests

## 📧 Support

If you encounter issues:
1. Check the Troubleshooting section
2. Verify all dependencies are installed
3. Ensure camera permissions are granted
4. Check that the model is trained properly

---

**Happy Gesturing! 🖐️✨**

Made with ❤️ using TensorFlow, MediaPipe, and OpenCV
