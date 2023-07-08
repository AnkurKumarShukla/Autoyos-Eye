import cv2
import time
import matplotlib.pyplot as plt
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 10

start_time = 0
end_time= start_time+ 10
EYE_CLOSED_COUNTER = 0  # Initialize EYE_CLOSED_COUNTER variable

def start_capture():
    global start_time, end_time, EYE_CLOSED_COUNTER
    start_time = time.time()
    end_time = start_time + 120  # Capture data for 2 minutes
    EYE_CLOSED_COUNTER = 0  # Reset EYE_CLOSED_COUNTER
    capture_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

# Rest of the code remains the same


def stop_capture():
    global end_time
    end_time = time.time()
    capture_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)

def save_data():
    filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
    if filepath:
        data = {"Time (s)": time_list, "EAR": ear_list, "Blink Type": blink_types}
        df = pd.DataFrame(data)
        df.to_excel(filepath, index=False)
        print("Data saved to", filepath)

def close_window():
    webcamFeed.release()
    output_video.release()
    cv2.destroyAllWindows()
    window.destroy()

def update_frame():
    global image
    (status, image) = webcamFeed.read()
    image = imutils.resize(image, width=800)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceDetector(grayImage, 0)

    for face in faces:
        faceLandmarks = landmarkFinder(grayImage, face)
        faceLandmarks = face_utils.shape_to_np(faceLandmarks)

        leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
        rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(image, [leftEyeHull], -1, (255, 0, 0), 2)
        cv2.drawContours(image, [rightEyeHull], -1, (255, 0, 0), 2)

        if ear < MINIMUM_EAR:
            EYE_CLOSED_COUNTER += 1
        else:
            EYE_CLOSED_COUNTER = 0

        ear_list.append(ear)
        time_list.append(time.time() - start_time)

        if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
            blink_types.append("Full blink")
        elif EYE_CLOSED_COUNTER > 0:
            blink_types.append("Partial blink")
        else:
            blink_types.append("No blink")

        cv2.putText(image, "EAR: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", image)
    cv2.waitKey(1)

    if time.time() < end_time:
        window.after(1, update_frame)
    else:
        save_data()

# Initialize the GUI window
window = tk.Tk()
window.title("Eye Blink Detection")
window.geometry("400x200")

# Create buttons
capture_button = tk.Button(window, text="Start Capture", command=start_capture)
capture_button.pack(pady=10)

stop_button = tk.Button(window, text="Stop Capture", command=stop_capture, state=tk.DISABLED)
stop_button.pack(pady=5)

exit_button = tk.Button(window, text="Exit", command=close_window)
exit_button.pack(pady=10)

# Initialize variables
ear_list = []
time_list = []
blink_types = []
EYE_CLOSED_COUNTER = 0

# Load the face detection model
FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
faceDetector = dlib.get_frontal_face_detector()
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)
webcamFeed = cv2.VideoCapture(0)

(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start the GUI event loop
window.after(1, update_frame)
window.mainloop()
