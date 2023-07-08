


import cv2
import time
import matplotlib.pyplot as plt
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import pandas as pd
import os
import datetime


output_folder = "data"
os.makedirs(output_folder, exist_ok=True)
def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear
faceDetector = dlib.get_frontal_face_detector()
# Load the face landmark predictor
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
# Initialize the video capture
ear_list = []
time_list = []
blink_types = []
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

video_path = 'path/input_vedio/test_vedio2.mp4'
cap = cv2.VideoCapture(video_path)
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Get the frame rate (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Generate filenames with current date and timestamp
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_filename = os.path.join(output_folder, f"video/output_video_{current_datetime}.mp4")
excel_filename = os.path.join(output_folder, f"excel/eye_data_{current_datetime}.xlsx")
# Create an empty DataFrame to store EAR values with time
ear_data = pd.DataFrame(columns=['Time', 'EAR'])
start_time = time.time()
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Break the loop if the video has ended
    if not ret:
        break
    
    # Convert the frame to grayscale
    # image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faceDetector = dlib.get_frontal_face_detector()
    faces = faceDetector(gray, 0)
    
    # Loop over the detected faces
    for face in faces:
        faceLandmarks = landmarkFinder(gray, face)
        faceLandmarks = face_utils.shape_to_np(faceLandmarks)

        leftEye = faceLandmarks[leftEyeStart:leftEyeEnd]
        rightEye = faceLandmarks[rightEyeStart:rightEyeEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        
        elapsed_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        print(elapsed_time)
        # Append the EAR value with the current time to the DataFrame
        ear_data = pd.concat([ear_data, pd.DataFrame({'Time': [elapsed_time], 'EAR': [ear]})], ignore_index=True)
        
        # Draw contours around the eyes
        left_eye_hull = cv2.convexHull(leftEye)
        right_eye_hull = cv2.convexHull(rightEye)
        
        cv2.drawContours(frame, [left_eye_hull], -1, (255, 0, 0), 2)
        cv2.drawContours(frame, [right_eye_hull], -1, (255, 0, 0), 2)
        
        # Classify the blink type
        if ear < 0.13:
            blink_types.append("Full blink")
        elif 0.13 <= ear < 0.2:
            blink_types.append("Partial blink")
        elif ear > 0.25:
            blink_types.append("Open eye")
        else:
            blink_types.append("Unsure")

        print("adding to list ")
        ear_list.append(ear)
        time_list.append(elapsed_time)
    # Display the frame
    cv2.imshow('Frame', frame)
     cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
       
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# this can be genetrated as a affect of new model 
# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Save the EAR data to an Excel file
# Save data to Excel
data = {"Time (s)": time_list, "EAR": ear_list, "Blink Type": blink_types}
df = pd.DataFrame(data)
df.to_excel(excel_filename, index=False)
print("Excel file saved to:", excel_filename)

# print("Real-time blink rate: {:.2f} blinks per second".format(blink_rate))
# ear_data.to_excel('ear_data.xlsx', index=False)