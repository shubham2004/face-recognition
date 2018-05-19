import cv2
i=0
import numpy as np
from PIL import Image
name=[]
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        print(Id)
        print(conf)
        cv2.putText(im,str(Id),(x,y-10),font,0.55,(0,255,0),1)
    if conf>45:
        cv2.imshow('im',im) 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
