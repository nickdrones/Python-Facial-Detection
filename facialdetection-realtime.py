import sys
import cv2

#import the cascade file for faces
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#initialize the webcam for facial recognition as the default video capture device
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #convert frame to greyscale to search for faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #create a list of the visible faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle with a dot in the middle around the visible faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.circle(frame,(x+(w/2),y+(h/2)), 5, (0,0,255), -1)

        #print the X and Y coordinates for the middle of the face
        print ("X: {}     Y: {} /n".format(x+(w/2),y+(h/2)))

    # Display the resulting frame with the overlay
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

#(x+w, y+h)
