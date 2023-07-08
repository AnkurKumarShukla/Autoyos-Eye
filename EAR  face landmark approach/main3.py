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

def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear

FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  
MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 30

faceDetector = dlib.get_frontal_face_detector()
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)
webcamFeed = cv2.VideoCapture(0)

(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
ear_list = []
time_list = []
blink_types = []

EYE_CLOSED_COUNTER = 0

frame_width = int(webcamFeed.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(webcamFeed.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the 'data' directory if it doesn't exist
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# Generate filenames with current date and timestamp
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_filename = os.path.join(output_folder, f"video/output_video_{current_datetime}.mp4")
excel_filename = os.path.join(output_folder, f"excel/eye_data_{current_datetime}.xlsx")

# Create the 'video' and 'excel' directories inside the 'data' directory if they don't exist
os.makedirs(os.path.dirname(video_filename), exist_ok=True)
os.makedirs(os.path.dirname(excel_filename), exist_ok=True)

output_video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

start_time = time.time()
end_time = start_time + 30  # Capture data for 10 seconds

try:
    while time.time() < end_time:
        (status, image) = webcamFeed.read()
        output_video.write(image)

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

            if ear < 0.13:
                blink_types.append("Full blink")
            elif 0.13 <= ear < 0.2:
                blink_types.append("Partial blink")
            elif ear > 0.25:
                blink_types.append("Open eye")
            else:
                blink_types.append("Unsure")

            cv2.putText(image, "EAR: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            ear_list.append(ear)
            time_list.append(time.time() - start_time)

        cv2.imshow("Frame", image)
        cv2.waitKey(1)

    # Calculate real-time blink rate
    blink_count = len([blink_type for blink_type in blink_types if blink_type == "Full blink"])
    total_time = time_list[-1] - time_list[0]
    blink_rate = blink_count / total_time

    # Save data to Excel
    data = {"Time (s)": time_list, "EAR": ear_list, "Blink Type": blink_types}
    df = pd.DataFrame(data)
    df.to_excel(excel_filename, index=False)
    print("Excel file saved to:", excel_filename)

    print("Real-time blink rate: {:.2f} blinks per second".format(blink_rate))

except Exception as e:
    print(e)

webcamFeed.release()
output_video.release()
cv2.destroyAllWindows()
