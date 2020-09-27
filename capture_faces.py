import numpy as np
import cv2
import pickle
import os

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

count = 0
person_name = input("Enter your name: ")
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    for v, k in og_labels.items():
        count += 1
og_labels[person_name] = count

for v, k in og_labels.items():
    print(v, k)

dirname = "./images/" + person_name
os.mkdir(dirname)
cap = cv2.VideoCapture(0)

img_no = 1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # for (x, y, w, h) in faces:
    # print(x,y,w,h)
    # roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
    #roi_color = frame[y:y+h, x:x+w]

    img_item = dirname + "/" + str(img_no) + ".jpg"

    cv2.imwrite(img_item, frame)

    # color = (255, 0, 0) #BGR 0-255
    #stroke = 2
    #end_cord_x = x + w
    #end_cord_y = y + h
    #cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    #subitems = smile_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in subitems:
    #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # Display the resulting frame
    img_no = img_no + 1
    cv2.imshow('frame', frame)
    if img_no == 200:
        break
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
