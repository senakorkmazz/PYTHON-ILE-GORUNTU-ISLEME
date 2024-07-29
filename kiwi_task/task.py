import cv2
import numpy as np

img = cv2.imread('input.jpg')
copy = img.copy()
hsv = cv2.cvtColor(copy,cv2.COLOR_BGR2HSV)

lower = np.array([25,50,0])
upper = np.array([45,255,255])

mask = cv2.inRange(hsv,lower,upper)

kernel = np.ones((5,5),dtype=np.uint8)

result = cv2.erode(mask,kernel,iterations=3)
result = cv2.dilate(mask,kernel,iterations=5)

result = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
result = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

mask_img = cv2.bitwise_and(copy,copy,mask=result)

def autoCanny(temp,sigma=0.33):
    median = np.median(temp)
    lower = int(max(0,(1-sigma)*median))
    upper = int(min(255, (1 + sigma) * median))
    canny = cv2.Canny(temp,lower,upper)
    return canny

canny = autoCanny(result)

contours, _ = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    if area>450:
        cv2.drawContours(mask_img,[hull],-1,(255,255,255),-1)
    elif area<450:
        cv2.drawContours(mask_img, [hull], -1, (0, 0, 0), -1)

cv2.imshow('original',img)
cv2.imshow('result',mask_img)

cv2.waitKey(0)
cv2.destroyAllWindows()