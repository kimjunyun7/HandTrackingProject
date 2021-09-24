import cv2
import mediapipe as mp
import time

# Webcam 1
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0  # Previous time
cTime = 0  # Current time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # If a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # handLms == hand landmarks, single hand ex) hand num 1, num 2
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)  # Position
                # print(id, cx, cy)
                if id == 0:  # 0 == First landmark
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # (cam image, single hand, draw connections)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # Show fps on cam screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
