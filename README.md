# HandGestureSystemControl

🚀 Hand Gesture System Controller
This project provides a touchless interface for your computer, allowing you to control the mouse and system functions using hand gestures. It leverages MediaPipe for landmark detection and TensorFlow for gesture classification.

🛠️ Prerequisites
Before running the project, ensure you have the following installed:

Python 3.10+ (Your environment is currently running 3.12.9).

Required Libraries: Install the necessary packages using the provided requirements file.

Bash
pip install -r requirements.txt
Missing Dependencies: Based on the latest tests, manually install these if they are missing:

Bash
pip install pandas scikit-learn seaborn
📂 Core Project Files
test_installation.py: A diagnostic script to verify your camera, dependencies, and project structure are ready.


collect_data.py: Used to record hand landmark coordinates for training custom gestures.

train_model.ipynb: A Jupyter notebook to train the deep learning model on your collected data.

gesture_controller.py: The main entry point that runs the real-time gesture recognition.

config.py: Contains configurable settings for camera resolution, smoothing, and gesture mappings.

🎮 Getting Started
Verify Environment: Run python test_installation.py to ensure your webcam and libraries are functional.


Collect Data: Execute python collect_data.py and perform the gestures prompted on screen.

Train the Model: Open train_model.ipynb and run all cells to generate models/gesture_model.h5.

Run Controller: Start the system by running python gesture_controller.py.

⌨️ In-App Controls
'q': Quit the application.

'p': Pause or resume tracking.

'm': Toggle the mini-preview window.