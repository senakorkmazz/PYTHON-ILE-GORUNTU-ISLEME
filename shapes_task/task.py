import cv2
import numpy as np

img = cv2.imread('input.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
ret,thres = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:

    area = cv2.contourArea(contour)
    if area<50000:
        M = cv2.moments(contour)
        cX = int(M['m10'] // M['m00'])
        cY = int(M['m01'] // M['m00'])

        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True), True)
        cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)
        if len(approx) == 3:
            cv2.putText(img,'TRIANGLE',(cX,cY),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        elif len(approx) == 4:
            cv2.putText(img,'SQUARE',(cX,cY),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        elif len(approx) == 5:
            cv2.putText(img,'PENTAGON',(cX,cY),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        else:
            cv2.putText(img,'CIRCLE',(cX,cY),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()