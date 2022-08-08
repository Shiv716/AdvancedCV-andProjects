import cv2 as cv
import numpy as np
import time
from DetectionModules import HandTrackModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# WEB-CAM Dimensions:
wCam, hCam = 480, 720

# pycaw initialisations:
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
print(volume.GetVolumeRange())
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

# Camera initialisations:
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(detectConf=0.7)

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        # print(lmlist[4], lmlist[8])

        # Co-ordinates for drawing the circle at the tip of index and thumb:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        # Point for drawing centre circle to linear figure:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Drawing the circle:
        cv.circle(img, (x1, y1), 10, (255, 0, 0), cv.FILLED)
        cv.circle(img, (x2, y2), 10, (255, 0, 0), cv.FILLED)
        # Joining the circles:
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # Drawing the circle at centre using the points above:
        cv.circle(img, (cx, cy), 8, (255, 0, 0), cv.FILLED)

        # Getting the distance between two tips:
        dist = math.hypot(x2 - x1, y2 - y1)
        print(dist)

        if dist <= 50:
            cv.circle(img, (cx, cy), 8, (0, 0, 255), cv.FILLED)
        if dist >= 200:
            cv.circle(img, (cx, cy), 8, (0, 255, 0), cv.FILLED)

        # Volume Controls:
        vol = np.interp(dist, [50, 300], [minVol, maxVol])
        volBar = np.interp(dist, [50, 300], [400, 150])
        volPer = np.interp(dist, [50, 300], [0, 100])
        vol.SetMasterVolumeLevel(vol, None)

    # Making Volume Bar on Screen:
    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv.FILLED)
    cv.putText(img, f'{int(volPer)}%', (40, 450), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'fps: {int(fps)}', (30, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv.imshow('img', img)
    cv.waitKey(1)
