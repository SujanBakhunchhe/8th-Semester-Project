import cv2 as cv
import numpy as np
import os
import math
import cvzone
import pickle
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
from keras_facenet import FaceNet
import tkinter as tk
from tkinter import IntVar, Checkbutton

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize FaceNet
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
svm_model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Initialize YOLO models
uniform_model = YOLO('student_model.pt')
person_model = YOLO('person.pt')
classnames = ['Shirt', 'Pant', 'Tie', 'Person']

# Initialize video capture
cap = cv.VideoCapture(0)  # Change to '2.mp4' if you want to use the video file

# Define HSV range for blue color
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
threshold = 500  # Minimum number of blue pixels to consider as detection

# Function to process frame
def process_frame(frame):
    frame = cv.resize(frame, (640, 480))
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Face detection
    if detect_face.get():
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        for x, y, w, h in faces:
            img = rgb_img[y:y + h, x:x + w]
            img = cv.resize(img, (160, 160))  # 1x160x160x3
            img = np.expand_dims(img, axis=0)
            ypred = facenet.embeddings(img)
            face_name = svm_model.predict(ypred)
            final_name = encoder.inverse_transform(face_name)[0]
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(frame, str(final_name), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv.LINE_AA)

    # Uniform detection
    if detect_uniform.get():
        uniform_result = uniform_model(frame, stream=True)
        for info in uniform_result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50 and classnames[Class] in ['Shirt', 'Pant', 'Tie']:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Extract the region of interest
                    roi = frame[y1:y2, x1:x2]

                    # Convert ROI to HSV color space
                    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

                    # Threshold the HSV image to get only blue colors
                    mask = cv.inRange(hsv_roi, lower_blue, upper_blue)

                    # Count the number of blue pixels
                    blue_pixel_count = cv.countNonZero(mask)

                    # If the blue pixel count exceeds the threshold, draw the bounding box
                    if blue_pixel_count > threshold:
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 - 10], scale=0.8,
                                           thickness=2)

    # Person detection
    if detect_person.get():
        person_result = person_model(frame, stream=True)
        for info in person_result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{classnames[-1]} {confidence}%', [x1 + 8, y1 - 10], scale=0.8, thickness=2)

    return frame

def start_detection():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        cv.imshow("Face, Uniform, and Person Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# Create UI
root = tk.Tk()
root.title("Object Detection Selection")

detect_face = IntVar()
detect_uniform = IntVar()
detect_person = IntVar()

Checkbutton(root, text="Detect Faces", variable=detect_face).pack()
Checkbutton(root, text="Detect Uniforms", variable=detect_uniform).pack()
Checkbutton(root, text="Detect Persons", variable=detect_person).pack()

start_button = tk.Button(root, text="Start Detection", command=start_detection)
start_button.pack()

root.mainloop()
