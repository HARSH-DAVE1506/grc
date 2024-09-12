import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import serial
import threading
import time
import random

# Serial communication settings
serial_port = '/dev/ttymxc3'
baud_rate = 115200

GESTURE_LABELS = {
    0: "Unknown",
    1: "Closed_Fist",
    2: "Open_Palm",
    3: "Pointing_Up",
    4: "Thumb_Down",
    5: "Thumb_Up",
    6: "Victory",
    7: "ILoveYou"
}

ZERO_COMMAND = {"T": 133, "X": 0, "Y": 0, "SPD": 0, "ACC": 0}  # Zero position

RANDOM_COMMANDS = [
    {"T": 133, "X": 60, "Y": 45, "SPD": 0, "ACC": 0},
    {"T": 133, "X": -60, "Y": -45, "SPD": 0, "ACC": 0},
    {"T": 133, "X": 30, "Y": -30, "SPD": 0, "ACC": 0},
    {"T": 133, "X": -30, "Y": 30, "SPD": 0, "ACC": 0}
]

COMMANDS = {
    "Open_Palm": {"T": 132, "IO4": 255, "IO5": 255},  # Turn on LED
    "Closed_Fist": {"T": 132, "IO4": 0, "IO5": 0},    # Turn off LED
    "ILoveYou": {"T": 133, "X": -90, "Y": -30, "SPD": 0, "ACC": 0},  # Shy (turn left, look down)
    "Thumb_Up": {"T": 133, "X": 0, "Y": 180, "SPD": 0, "ACC": 0},  # all up
    "Thumb_Down": {"T": 133, "X": 0, "Y": -30, "SPD": 0, "ACC": 0}, # all down
    "Victory": {"T": 133, "X": 180, "Y": 0, "SPD": 0, "ACC": 0}, # round 180
}

# Initialize the gesture recognition model
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Initialize serial communication
try:
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
except serial.SerialException as e:
    print(f"Failed to open serial port: {e}")
    ser = None

# Cache the last sent command to avoid redundant serial writes
last_sent_command = None

# Event flag to pause random commands after gesture detection
gesture_detected_event = threading.Event()

def send_serial_command(command):
    global last_sent_command
    if ser is None or command == last_sent_command:
        return  # Do nothing if serial connection is unavailable or command is unchanged
    json_command = json.dumps(command)
    ser.write(f"{json_command}\n".encode())
    last_sent_command = command
    time.sleep(0.1)  # Small delay to ensure command is sent

def send_zero_command():
    send_serial_command(ZERO_COMMAND)
    print("Zero command sent")

def random_command_worker():
    """Thread to send random commands every 5 seconds unless a gesture is detected."""
    while True:
        if not gesture_detected_event.is_set():
            random_command = random.choice(RANDOM_COMMANDS)
            send_serial_command(random_command)
            print(f"Random command sent: {random_command}")
            threading.Timer(2.0, send_zero_command).start()  # Zero command after 2 seconds
        gesture_detected_event.wait(5)  # Wait for 5 seconds or until gesture is detected

# Start random command worker thread
random_command_thread = threading.Thread(target=random_command_worker)
random_command_thread.daemon = True
random_command_thread.start()

def main():
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Set a lower resolution for the camera to reduce CPU load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        # Read a frame from the camera
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera. Retrying...")
            time.sleep(1)
            continue

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            # Recognize gestures
            recognition_result = recognizer.recognize(image)

            # Process the result
            if recognition_result.gestures:
                top_gesture = recognition_result.gestures[0][0]
                gesture_name = top_gesture.category_name
                print(f"Detected gesture: {gesture_name}")

                if gesture_name in COMMANDS:
                    gesture_command = COMMANDS[gesture_name]
                    send_serial_command(gesture_command)
                    print(f"Gesture command sent: {gesture_command}")

                    # Send zero command after 2 seconds if necessary
                    if gesture_name in ["ILoveYou", "Thumb_Up", "Thumb_Down", "Victory", "Pointing_Up"]:
                        threading.Timer(2.0, send_zero_command).start()

                    # Pause random commands for 10 seconds after gesture detection
                    gesture_detected_event.set()
                    threading.Timer(10.0, gesture_detected_event.clear).start()

        except Exception as e:
            print(f"Error during gesture recognition: {e}")

        # Display the frame (optional, can be disabled for more performance)
        #cv2.imshow('Gesture Recognition', frame)

        # Exit on key press (escape)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the camera and close the serial connection
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == "__main__":
    main()
