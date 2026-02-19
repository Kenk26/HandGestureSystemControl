"""
Standalone Data Collection Script
Use this if you prefer not to use Jupyter Notebook for data collection
"""

import cv2
import os
import mediapipe as mp
import sys

# Gesture definitions
GESTURES = {
    0: 'palm',
    1: 'index',
    2: 'peace',
    3: 'fist',
    4: 'thumb_pinky',
    5: 'okay'
}

GESTURE_DESCRIPTIONS = {
    'palm': 'Open palm - all fingers extended',
    'index': 'Only index finger pointing up',
    'peace': 'Peace sign - index and middle finger up',
    'fist': 'Closed fist - all fingers down',
    'thumb_pinky': 'Shaka sign - thumb and pinky extended',
    'okay': 'OK sign - thumb and index forming circle'
}

IMG_SIZE = 128
SAMPLES_PER_GESTURE = 500
DATA_DIR = '../data/raw'

class DataCollector:
    """
    Collect gesture data from webcam
    """
    
    def __init__(self):
        """
        Initialize data collector
        """
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Create data directory
        os.makedirs(DATA_DIR, exist_ok=True)
        
        print("Data Collector initialized!")
    
    def collect_gesture(self, gesture_name, gesture_id, num_samples=500):
        """
        Collect samples for a single gesture
        """
        # Create directory for this gesture
        gesture_dir = os.path.join(DATA_DIR, gesture_name)
        os.makedirs(gesture_dir, exist_ok=True)
        
        # Count existing samples
        existing_files = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
        start_count = len(existing_files)
        
        print(f"\n{'='*60}")
        print(f"Collecting: {gesture_name.upper()}")
        print(f"{'='*60}")
        print(f"Description: {GESTURE_DESCRIPTIONS[gesture_name]}")
        print(f"Existing samples: {start_count}")
        print(f"Target: {num_samples} samples")
        print(f"\nInstructions:")
        print("  - Position your hand clearly in front of the camera")
        print("  - Press SPACE to start/stop capturing")
        print("  - Move your hand slightly while capturing for variation")
        print("  - Press 'q' to finish early and move to next gesture")
        print("  - Press 'r' to reset and delete all samples for this gesture")
        print(f"{'='*60}\n")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        count = start_count
        capturing = False
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from camera")
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    if capturing:
                        # Get bounding box
                        h, w, c = frame.shape
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        
                        x_min = int(min(x_coords) * w) - 20
                        x_max = int(max(x_coords) * w) + 20
                        y_min = int(min(y_coords) * h) - 20
                        y_max = int(max(y_coords) * h) + 20
                        
                        # Clip to frame boundaries
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(w, x_max)
                        y_max = min(h, y_max)
                        
                        # Extract hand region
                        hand_roi = frame[y_min:y_max, x_min:x_max]
                        
                        if hand_roi.size > 0:
                            # Resize and save
                            hand_roi_resized = cv2.resize(hand_roi, (IMG_SIZE, IMG_SIZE))
                            img_path = os.path.join(
                                gesture_dir, 
                                f"{gesture_name}_{count}.jpg"
                            )
                            cv2.imwrite(img_path, hand_roi_resized)
                            count += 1
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                        (0, 255, 0), 2)
            
            # Display UI
            status = "CAPTURING" if capturing else "Ready - Press SPACE"
            status_color = (0, 255, 0) if capturing else (0, 255, 255)
            
            # Background overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            # Text
            cv2.putText(frame, f"Gesture: {gesture_name.upper()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Status: {status}", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Collected: {count}/{num_samples}", 
                       (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Progress bar
            progress = count / num_samples
            bar_width = int(640 * progress)
            cv2.rectangle(frame, (0, 110), (bar_width, 120), (0, 255, 0), -1)
            
            cv2.imshow('Data Collection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                capturing = not capturing
                status_msg = "Started" if capturing else "Stopped"
                print(f"  {status_msg} capturing - Count: {count}/{num_samples}")
            
            elif key == ord('q'):
                print(f"  Stopped early - Collected {count} samples")
                break
            
            elif key == ord('r'):
                print(f"  Resetting gesture '{gesture_name}'...")
                # Delete all files
                for f in os.listdir(gesture_dir):
                    os.remove(os.path.join(gesture_dir, f))
                count = 0
                print(f"  Reset complete - Start collecting again")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Collected {count} samples for '{gesture_name}'")
        
        return count
    
    def collect_all(self):
        """
        Collect data for all gestures
        """
        print("\n" + "="*60)
        print("HAND GESTURE DATA COLLECTION")
        print("="*60)
        print(f"\nTotal gestures: {len(GESTURES)}")
        print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
        print(f"Total samples to collect: {len(GESTURES) * SAMPLES_PER_GESTURE}")
        
        total_collected = 0
        
        for gesture_id, gesture_name in GESTURES.items():
            input(f"\nPress ENTER to start collecting '{gesture_name}' gesture...")
            
            count = self.collect_gesture(gesture_name, gesture_id, SAMPLES_PER_GESTURE)
            total_collected += count
        
        print("\n" + "="*60)
        print("DATA COLLECTION COMPLETE!")
        print("="*60)
        print(f"\nTotal samples collected: {total_collected}")
        print(f"Data saved in: {DATA_DIR}")
        print("\nNext steps:")
        print("  1. Review collected data")
        print("  2. Train model using Jupyter notebook")
        print("  3. Test with gesture_controller.py")
        print("="*60)
    
    def close(self):
        """
        Clean up resources
        """
        self.hands.close()

def main():
    """
    Main function
    """
    print("\n" + "="*60)
    print("STANDALONE DATA COLLECTOR")
    print("="*60)
    
    # Check camera
    print("\nChecking camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: Cannot open camera")
        print("  Make sure no other application is using the camera")
        sys.exit(1)
    cap.release()
    print("✓ Camera OK")
    
    # Initialize collector
    collector = DataCollector()
    
    # Menu
    print("\nOptions:")
    print("  1. Collect all gestures (recommended)")
    print("  2. Collect specific gesture")
    print("  3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        collector.collect_all()
    
    elif choice == '2':
        print("\nAvailable gestures:")
        for i, (gid, gname) in enumerate(GESTURES.items(), 1):
            print(f"  {i}. {gname} - {GESTURE_DESCRIPTIONS[gname]}")
        
        gesture_num = input("\nEnter gesture number: ").strip()
        
        try:
            gesture_num = int(gesture_num) - 1
            if 0 <= gesture_num < len(GESTURES):
                gesture_name = GESTURES[gesture_num]
                num_samples = input(f"Number of samples (default {SAMPLES_PER_GESTURE}): ").strip()
                num_samples = int(num_samples) if num_samples else SAMPLES_PER_GESTURE
                
                collector.collect_gesture(gesture_name, gesture_num, num_samples)
            else:
                print("Invalid gesture number")
        except ValueError:
            print("Invalid input")
    
    elif choice == '3':
        print("Exiting...")
    
    else:
        print("Invalid choice")
    
    collector.close()
    print("\nGoodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
