import cv2 as cv
import numpy as np 
import os

haar_cascade = cv.CascadeClassifier('haar_face.xml')

DIR = r'C:\Users\kanek\Documents\opencv\Resources\Faces\train'

p = []
for i in os.listdir(DIR):
    p.append(i)

# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\kanek\Documents\opencv\Resources\Faces\val\madonna\3.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {p[label]} with a confidence of {confidence}')

    cv.putText(img, str(p[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)