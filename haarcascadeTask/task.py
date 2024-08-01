import cv2

video = cv2.VideoCapture('input.mp4')
cascade = cv2.CascadeClassifier('frontallface.txt')

cv2.namedWindow('Face Detector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Detector', 540, 960)

while True:

    ret, frame = video.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(gray_frame,1.3,4)

    for x,y,w,h in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Face Detector', frame)


video.release()
cv2.destroyAllWindows()

