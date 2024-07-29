import time

import cv2
import numpy as np
import datetime

start = time.time()

def count_fruits():
    img = cv2.imread("input.jpg")
    copy = img.copy()
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, thres = cv2.threshold(blur, 245, 350, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), dtype=np.uint8)
    result = cv2.erode(thres, kernel, iterations=2)
    result = cv2.dilate(thres, kernel, iterations=1)

    def autoCanny(blur, sigma=0.33):
        median = np.median(blur)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        canny = cv2.Canny(blur, lower, upper)
        return canny

    contours, _ = cv2.findContours(autoCanny(result), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rows,cols = 4, 3
    height, width = img.shape[:2]
    cell_height, cell_width = height // rows, width // cols

    grid = [[[] for _ in range(cols)] for _ in range(rows)]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            grid_row = cY // cell_height
            grid_col = cX // cell_width
            grid[grid_row][grid_col].append((cX, cY, contour))

    # Konturları numaralandır ve çiz
    i = 1
    for row in range(rows):
        for col in range(cols):
            for cX, cY, contour in sorted(grid[row][col], key=lambda x: (x[0], x[1])):
                cv2.drawContours(copy, [contour], -1, (0, 0, 0), 3)
                cv2.putText( copy,str(i), (cX - 15, cY - 3),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0),2)
                i += 1

    cv2.imshow("original", img)
    cv2.imshow("contour", copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


count_fruits()
finish = time.time()

count = finish-start
print(count)