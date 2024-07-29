import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('input1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

equalization = cv2.equalizeHist(gray)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(gray,cmap='gray')

plt.subplot(1,2,2)
plt.title('Synchronized')
plt.imshow(equalization,cmap='gray')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
