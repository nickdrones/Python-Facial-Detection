import numpy as np
import cv2

#import the cascade file for faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#set the image to find faces
img = cv2.imread('group.jpg')

#convert image to greyscale to search for faces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create a list of the visible faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#draw a rectangle around each visible face
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

#resize window to custom size (ex: 900 wide by 600 high) and display image
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 900,600)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
