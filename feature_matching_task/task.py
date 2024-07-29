import cv2
import numpy as np

img1 = cv2.imread('input1.jpeg')
img1 = cv2.pyrDown(img1)
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('input2.jpeg')
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(gray1,None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)

matching = bf.match(descriptors1,descriptors2)
matching = sorted(matching, key = lambda x: x.distance)

result = cv2.drawMatches(img1,keypoints1,img2,keypoints2, matching[:50], None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
