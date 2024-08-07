import cv2
import numpy as np

img = cv2.imread('input.jpg')

mask = np.zeros((img.shape[:2]), np.uint8)
bgM = np.zeros((1,65),np.float64)
fgM = np.zeros((1,65),np.float64)

rectangle = cv2.selectROI('select roi',img)

cv2.grabCut(img,mask,rectangle,bgM,fgM,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where(((mask==0) | (mask==2)),0,1).astype('uint8')
result = img * mask2[:,:,np.newaxis]

cv2.imshow('result',result)

cv2.waitKey(0)
cv2.destroyAllWindows()