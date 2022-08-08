import cv2 as cv
import time
import mediapipe as mp


class faceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectCon = minDetectCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh()
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec, self.drawSpec)

                face = []
                for id, lm in enumerate(facelms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # Printing id of each point on the face:
                    cv.putText(img, str(id), (x, y), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

                    face.append([x, y])
                faces.append(face)

        return img, faces


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture('Movie.mov')
    detector = faceMeshDetector(maxFaces=1)

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(faces[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv.imshow('img', img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
