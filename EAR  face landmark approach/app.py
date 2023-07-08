import tkinter as tk
import cv2
import time
import matplotlib.pyplot as plt
import dlib
import imutils
from PIL import ImageTk

from imutils import face_utils
from scipy.spatial import distance as dist
from PIL import Image

class EyeTrackerApp:
    def __init__(self, master):
        self.master = master
        master.title("Eye Tracker App")
        
        self.label = tk.Label(master, text="Eye Aspect Ratio")
        self.label.pack()
        
        self.canvas = tk.Canvas(master, width=800, height=600)
        self.canvas.pack()
        
        self.start_button = tk.Button(master, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = tk.Button(master, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.ear_label = tk.Label(master, text="")
        self.ear_label.pack()
        self.running = False
        self.webcamFeed = None
        self.ear_list = []
        self.time_list = []
        self.EYE_CLOSED_COUNTER = 0
        
        self.FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  
        self.MINIMUM_EAR = 0.2
        self.MAXIMUM_FRAME_COUNT = 10
        
        self.faceDetector = dlib.get_frontal_face_detector()
        self.landmarkFinder = dlib.shape_predictor(self.FACIAL_LANDMARK_PREDICTOR)

        (self.leftEyeStart, self.leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rightEyeStart, self.rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
    def start(self):
        self.running = True
        self.webcamFeed = cv2.VideoCapture(0)
        self.update()
        
    def stop(self):
        self.running = False
        self.webcamFeed.release()
        
    def update(self):
        if not self.running:
            return
        (status, image) = self.webcamFeed.read()
        image = imutils.resize(image, width=800)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fps = self.webcamFeed.get(cv2.CAP_PROP_FPS)
        print("Frame rate:", fps)
        faces = self.faceDetector(grayImage, 0)

        for face in faces:
            faceLandmarks = self.landmarkFinder(grayImage, face)
            faceLandmarks = face_utils.shape_to_np(faceLandmarks)

            leftEye = faceLandmarks[self.leftEyeStart:self.leftEyeEnd]
            rightEye = faceLandmarks[self.rightEyeStart:self.rightEyeEnd]

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(image, [leftEyeHull], -1, (255, 0, 0), 2)
            cv2.drawContours(image, [rightEyeHull], -1, (255, 0, 0), 2)

            if ear < self.MINIMUM_EAR:
                self.EYE_CLOSED_COUNTER += 1
            else:
                self.EYE_CLOSED_COUNTER = 0
            self.ear_list.append(ear)
            
            self.time_list.append(time.time())
            self.ear_label.configure(text=f"EAR: {ear:.2f}")
            self.time_list.append(time.time()) 
            # cv2.putText(image, "scanning...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,0, 255), 2)          
            if self.EYE_CLOSED_COUNTER >= self.MAXIMUM_FRAME_COUNT:
                cv2.putText(image, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # plt.clf()
        # plt.plot(self.time_list, self.ear_list)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Eye Aspect Ratio")
        # plt.draw()

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 600))
        img = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.master.after(20, self.update)
        
    def eye_aspect_ratio(self, eye):
        a = dist.euclidean(eye[1], eye[5])
        b = dist.euclidean(eye[2], eye[4])
        c = dist.euclidean(eye[0], eye[3])
        ear = (a + b) / (2.0 * c)
        return ear
        
root = tk.Tk()
app = EyeTrackerApp(root)
root.mainloop()

           
