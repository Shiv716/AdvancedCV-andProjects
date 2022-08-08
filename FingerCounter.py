import cv2 as cv
import os
import time
from DetectionModules import HandTrackModule as htm

# Cam Dimensions:
wCam, hCam = 1040, 1080

cap = cv.VideoCapture(0)
# Setting , width and height to cam:
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'Fingers'
myList = os.listdir(folderPath)
myList.sort()
print(myList)
overlayList = []

for impath in myList:
    image = cv.imread(f'{folderPath}/{impath}')
    overlayList.append(image)
print(len(overlayList))

pTime = 0
detector = htm.handDetector(trackCon=0.75)
# Initial point of every finger:
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Making sure to detect all fingers through their tip points:
        fingers = []

        # For Right Hand:-
        fingers.clear()
        if lmList[1][1] > lmList[17][1]:

            # For thumb:-
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # For all fingers:-
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # print(fingers)

            # COUNTING THE FINGERS:-
            total_fingers = fingers.count(1)
            h, w, c = overlayList[total_fingers].shape
            img[0:h, 0:w] = overlayList[total_fingers]

            # Rectangle text of the finger count:
            cv.rectangle(img, (1000, 100), (1200, 465), (0, 255, 0), cv.FILLED)
            cv.putText(img, str(total_fingers), (1000, 375), cv.FONT_HERSHEY_COMPLEX,
                       10, (255, 0, 0), 25)

        # For Left Hand:-
        if lmList[3][1] < lmList[18][1]:
            fingers.clear()

            # For thumb:-
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # For all fingers:-
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # print(fingers)

            # COUNTING THE FINGERS:-
            total_fingers = fingers.count(1)
            h, w, c = overlayList[total_fingers].shape
            img[0:h, 0:w] = overlayList[total_fingers]

            # Rectangle text of the finger count:
            cv.rectangle(img, (1000, 100), (1200, 465), (0, 255, 0), cv.FILLED)
            cv.putText(img, str(total_fingers), (1000, 375), cv.FONT_HERSHEY_COMPLEX,
                       10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'fps:{int(fps)}', (700, 70), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 3)

    cv.imshow('Video', img)
    cv.waitKey(1)
