import cv2
import sys
import numpy as np
import os
import time


dir_name = "database"
person_name = sys.argv[1]
sys_path = os.path.join(dir_name, person_name)

if not os.path.isdir(sys_path):
    os.mkdir(sys_path)

casc_class = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
count = 0

while count<45:
    ret, frame = video_capture.read()
    image = cv2.flip(frame, 1, 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    minimage = cv2.resize(gray, (int(gray.shape[1]/4), int(gray.shape[0]/4)))
    faces = casc_class.detectMultiScale(minimage)
    faces = sorted(faces, key=lambda x: x[3])

    if faces:
        face1 = faces[0]
        (x,y,w,h) = [v*4 for v in face1]
        face = gray[y:y+h, x:x+w]
        face_original = cv2.resize(face, (112, 92))
        loc = sorted([int(a[:a.find('.')]) for a in os.listdir(sys_path) if a[0]!='.' ]+[0])[-1] + 1
        print('%s%s.png' % (sys_path, loc))
        cv2.imwrite('%s%s.png' % (sys_path, loc), face_original)
        cv2.rectangle(image, (x,y), (x+w,y+h),(0,255,0), thickness = 4)
        cv2.putText(image, person_name, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0))
        time.sleep(0.38)
        count+=1

    cv2.imshow('Opencv', image)
    key = cv2.waitKey(10)
    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
