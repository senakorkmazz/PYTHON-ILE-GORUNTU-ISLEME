import cv2
import numpy as np

img = cv2.imread('input.jpg')
img = cv2.pyrDown(img)
hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

red_lower1 = np.array([100,30,30])
red_upper1 = np.array([255,255,255])
red_mask1= cv2.inRange(hls,red_lower1,red_upper1)

red_lower2 = np.array([0,10,10])
red_upper2 = np.array([10,255,255])
red_mask2 = cv2.inRange(hls,red_lower2, red_upper2)

mask = cv2.bitwise_or(red_mask1,red_mask2)

kernel = np.ones((3,3),np.uint8)
mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)

mask = cv2.erode(mask, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=1)

hls[mask>0,0] = 135
purple_rose = cv2.cvtColor(hls,cv2.COLOR_HLS2BGR)

purple_rose[mask==0] = (0,0,0)

cv2.imshow('original', img)
cv2.imshow('purple rose',purple_rose)

cv2.waitKey(0)
cv2.destroyAllWindows()