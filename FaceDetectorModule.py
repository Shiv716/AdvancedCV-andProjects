import cv2 as cv
import mediapipe as mp
import time


class faceDetector():
    def __init__(self, minTrackCon=0.5):
        self.minTrackCon = minTrackCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=minTrackCon)

    def findfaces(self, img, draw=True, bboxs =True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs =[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                self.mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape

                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                bboxs.append([id, bbox, detection.score])

                # Fancy Draw on the rectangle:
                x, y, w, h = bbox
                x1, y1 = x+w, y+h

                # Drawing our own rectangle:
                cv.rectangle(img, bbox, (255, 0, 255), 2)

                # Customising the rectangle:
                # CORNER 1: (TOP LEFT)
                cv.line(img, (x, y), (x + 30, y), (255, 0, 255), 10)
                cv.line(img, (x, y), (x, y + 30), (255, 0, 255), 10)
                # CORNER 2:
                cv.line(img, (x1, y), (x1 - 30, y), (255, 0, 255), 10)
                cv.line(img, (x1, y), (x1, y + 30), (255, 0, 255), 10)
                # CORNER 3: (LOWER LEFT)
                cv.line(img, (x, y1), (x + 30, y1), (255, 0, 255), 10)
                cv.line(img, (x, y1), (x, y1 - 30), (255, 0, 255), 10)
                # CORNER 4:
                cv.line(img, (x1, y1), (x1 - 30, y1), (255, 0, 255), 10)
                cv.line(img, (x1, y1), (x1, y1 - 30), (255, 0, 255), 10)

                # Confidence display:
                cv.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                           cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        return img, bboxs

# -----


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture('Movie.mov')
    detector = faceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.findfaces(img)
        # print(bboxs)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv.imshow('img', img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
