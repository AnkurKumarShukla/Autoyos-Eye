import cv2
import time
import matplotlib.pyplot as plt
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist



def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear



FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  
MINIMUM_EAR = 0.2
MAXIMUM_FRAME_COUNT = 10

faceDetector = dlib.get_frontal_face_detector()
landmarkFinder = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)
webcamFeed = cv2.VideoCapture(0)

(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
ear_list = []
time_list = []


EYE_CLOSED_COUNTER = 0

try:
    while True:
        (status, image) = webcamFeed.read()
        image = imutils.resize(image, width=800)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fps = webcamFeed.get(cv2.CAP_PROP_FPS)
        print("Frame rate:", fps)
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
            time_list.append(time.time())
            cv2.putText(image, "EAR: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
                cv2.putText(image, "Drowsiness", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            plt.plot(time_list, ear_list)
            plt.xlabel('Time')
            plt.ylabel('Eye Opening Probability')
            plt.show()
     
        cv2.imshow("Frame", image)
        cv2.waitKey(1)
    # plt.plot(time_list, ear_list)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Eye Aspect Ratio')
    # plt.show(block=False)
        
except Exception as e:
    print(e)


