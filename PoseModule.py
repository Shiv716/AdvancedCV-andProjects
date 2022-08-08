import cv2 as cv
import time
import mediapipe as mp
import math


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                cv.circle(img, (cx, cy), 2, (255, 0, 0), cv.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True, draw_angle=True):
        # Getting the landmarks:
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Getting the angle:-
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        # print(f'The angle is: {int(angle)}')
        if angle < 0:
            angle += 360

        if draw:
            # Drawing lines:
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv.line(img, (x2, y2), (x3, y3), (0, 255, 0), 3)
            # Drawing circles:
            cv.circle(img, (x1, y1), 3, (0, 50, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv.circle(img, (x2, y2), 3, (0, 50, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv.circle(img, (x3, y3), 3, (0, 50, 255), cv.FILLED)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), 2)

        # Texting the angle:
        if draw_angle:
            cv.putText(img, str(int(angle)), (x2-50, y2+50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 50), 2)

        return angle


def main():
    pTime = 0
    cap = cv.VideoCapture('Movie.mov')
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        lmList = detector.findPosition(img)
        print(lmList[14])
        cv.circle(img, (lmList[14][1], lmList[14][2]), 7, (0, 0, 255), cv.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.imshow('image', img)
        cv.waitKey(1)


if __name__ == '__main__':
    main()
