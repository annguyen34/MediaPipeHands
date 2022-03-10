
import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 0: Setting cam
mp_hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands

while True:
    success, img = cam.read()
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(imgRBG)

    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hm in results.multi_hand_landmarks:
            for lm in hm.landmark:
                h, w, d = img.shape
                cx = int(lm.x*w)
                cy = int(lm.y*h)
                cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
            mp_drawing.draw_landmarks(img, hm, hands.HAND_CONNECTIONS)

    cv2.imshow("Testing", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
