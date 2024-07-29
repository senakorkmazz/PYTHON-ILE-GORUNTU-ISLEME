import cv2
import numpy as np
from matplotlib import pyplot as plt

img1= cv2.imread('input1.jpg')
img2= cv2.imread('input2.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

hist1 = cv2.calcHist(gray1,[0],None,[256],[0,256])
hist2 = cv2.calcHist(gray2,[0],None,[256],[0,256])

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img1)
plt.title('Original Image')

plt.subplot(1,2,2)
plt.xlabel('Pixel Values')
plt.ylabel('Frequency')
plt.title('Color Distribution Histogram')
plt.hist(hist1)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img2)
plt.title('Original Image')

plt.subplot(1,2,2)
plt.xlabel('Pixel Values')
plt.ylabel('Frequency')
plt.title('Color Distribution Histogram')
plt.hist(hist2)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()