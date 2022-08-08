import cv2 as cv
import time
import mediapipe as mp

cap = cv.VideoCapture('Movie.mov')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.75)

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            # Drawing our own rectangle:
            cv.rectangle(img, bbox, (255, 0, 255), 3)
            # Confidence display:
            cv.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                       cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'fps: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN,
               3, (0, 255, 0), 2)

    cv.imshow('img', img)
    cv.waitKey(10)
