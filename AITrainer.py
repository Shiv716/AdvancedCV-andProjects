import cv2 as cv
import time
from DetectionModules import PoseModule as pm
import numpy as np

detector = pm.poseDetector(trackCon=0.9)
cap = cv.VideoCapture('WorkoutVideo1.1.mp4')
count = 0
dir = 0
pTime = 0
while True:
    success, img = cap.read()

    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Right Arm:
        angle = detector.findAngle(img, 12, 14, 16, draw_angle=False)
        # Left Arm:
        # detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (68, 170), (100, 0))
        bar = np.interp(angle, (68, 170), (100, 650))
        print(angle, per)
        # initialising bar colour:-
        colour = (255, 0, 255)
        # Check for dumbbell curls;
        if per >= 70:
            colour = (0, 255, 0)

            if dir == 0:
                count += 0.5
                dir = 1
        if per <= 30:
            colour = (0, 0, 255)

            if dir == 1:
                count += 0.5
                dir = 0
        print(count)

        # Drawing the bar:-
        cv.rectangle(img, (1100, 100), (1175, 650), (255, 0, 255), 3)
        cv.rectangle(img, (1100, int(bar)), (1175, 650), colour, cv.FILLED)
        cv.putText(img, f'{int(per)}%', (1050, 85), cv.FONT_HERSHEY_PLAIN, 7, (0, 255, 0), 10)

    # Printing the count:-
    cv.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv.FILLED)
    cv.putText(img, str(int(count)), (45, 670), cv.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    # fps:-
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'fps{int(fps)}', (50, 100), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

    cv.imshow('img', img)
    cv.waitKey(1)
