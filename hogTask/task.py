import cv2
from skimage.feature import hog
from skimage import exposure

img = cv2.imread('input.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_,hogImage = hog(gray,visualize=True)
rescaled = exposure.rescale_intensity(hogImage,in_range=(0,10))

cv2.imshow('original',img)
cv2.imshow('result',rescaled)

cv2.waitKey(0)
cv2.destroyAllWindows()