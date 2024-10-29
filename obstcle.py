import cv2
import serial
import time

# Set up serial communication with Arduino
arduino = serial.Serial('COM3', 9600, timeout=1)  # replace 'COM3' with your Arduino port
time.sleep(2)  # wait for the connection to initialize

# Initialize OpenCV
cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def send_angle_to_servo(angle):
    arduino.write(f"{angle}\n".encode())
    time.sleep(0.05)  # small delay to allow the servo to move

angle = 90  # start at center position
send_angle_to_servo(angle)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces and adjust servo angle
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_center_x = x + w // 2
            frame_center_x = frame.shape[1] // 2

            # Adjust angle based on face position
            if face_center_x < frame_center_x - 30:
                angle -= 5
            elif face_center_x > frame_center_x + 30:
                angle += 5

            # Keep the angle within 0-180 degrees
            angle = max(0, min(180, angle))
            send_angle_to_servo(angle)

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
import cv2
import serial
import time

# Set up serial communication with Arduino
arduino = serial.Serial('COM3', 9600, timeout=1)  # replace 'COM3' with your Arduino port
time.sleep(2)  # wait for the connection to initialize

# Initialize OpenCV
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def send_angle_to_servo(angle):
    arduino.write(f"{angle}\n".encode())
    time.sleep(0.05)  # small delay to allow the servo to move

angle = 90  # start at center position
send_angle_to_servo(angle)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces and adjust servo angle
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_center_x = x + w // 2
            frame_center_x = frame.shape[1] // 2

            # Adjust angle based on face position
            if face_center_x < frame_center_x - 30:
                angle -= 5
            elif face_center_x > frame_center_x + 30:
                angle += 5

            # Keep the angle within 0-180 degrees
            angle = max(0, min(180, angle))
            send_angle_to_servo(angle)

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()
