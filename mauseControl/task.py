import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

screen_w , screen_h = pyautogui.size()

while True:

    ret,frame = video.read()
    frame = cv2.flip(frame,1)
    w,h,_ = frame.shape

    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    output = hands.process(rgb_frame)
    result = output.multi_hand_landmarks

    if result:
        for hand_landmark in result:
            index_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]


            index_finger_x = int(index_finger_tip.x * w)
            index_finger_y = int(index_finger_tip.y * h)

            screen_x = screen_w / w * index_finger_x
            screen_y = screen_h / h * index_finger_y

            pyautogui.moveTo(screen_x,screen_y)

            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            middle_finger_x = int(middle_finger_tip.x * w)
            middle_finger_y = int(middle_finger_tip.y * h)

            distance1 = (((index_finger_x - thumb_x) ** 2) + ((index_finger_y - thumb_y) ** 2) ** 0.5)
            distance2 = (((index_finger_x - middle_finger_x)**2)+((index_finger_y - middle_finger_y)**2)**0.5)

            if distance1<20:
                pyautogui.click()

            if distance2<20:
                pyautogui.rightClick()

            mp_drawing.draw_landmarks(frame,hand_landmark,mp_hands.HAND_CONNECTIONS)

    cv2.imshow('mouse control',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


