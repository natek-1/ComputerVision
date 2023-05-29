import mediapipe as mp
import time
import cv2

class FaceMeshDetector():
    def __init__(self, static_image_mode=False,
               max_num_faces=2,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.FaceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.mode,
                                                    max_num_faces=self.num_faces,
                                                    refine_landmarks=self.refine_landmarks,
                                                    min_detection_confidence=self.min_detection_confidence,
                                                    min_tracking_confidence=self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.results = None
    
    def findFaceMesh(self, img, draw=True):
        while True:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.FaceMesh.process(imgRGB)
            faces_info = []
            if self.results.multi_face_landmarks:
                for faceLms in self.results.multi_face_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                    #using enumerate to tell us what the id of the landmark is 
                    # which can be used to specify which specific landmark you want
                    face_info = []
                    for id, lm in enumerate(faceLms.landmark):
                        height, width, _ = img.shape
                        x,y = int(lm.x*width), int(lm.y* height)
                        # just to visually see which part of the face each id is in
                        #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
                        face_info.append((id, x,y))
                    faces_info.append(face_info)
            return faces_info
                


def main():
    cap = cv2.VideoCapture("FaceVideos/11.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    success = True
    while success:
        success, img = cap.read()
        faces = detector.findFaceMesh(img)
        cTime = time.time()
        if len(faces) != 0:
            print(faces)
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255, 0), 3)
        cv2.imshow("Test Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()