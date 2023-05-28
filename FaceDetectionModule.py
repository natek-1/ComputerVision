import mediapipe as mp
import cv2
import time


class FaceDetector():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_confidence = min_detection_confidence
        self.model = model_selection
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
    
    def findFaces(self, img, draw=True):
        #Getting the frame rate of the video
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                #mpDraw.draw_detection(img, detection)
                # using our own drawing and information to make a solid drawing of the face in the image
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h) , int(bboxC.width *w), int(bboxC.height*h)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    self.fancyDraw(img, bbox)
                    cv2.putText(img, f"Certainty: {int(detection.score[0]*100)}%",
                            (bbox[0],bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return bboxs
    
    def fancyDraw(self, img, bbox, length=70, thickness=10):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (255, 0, 255), 2)
        ## top left
        cv2.line(img, (x,y), (x+length, y), (255, 0, 255), thickness=thickness)
        cv2.line(img, (x, y), (x, y+length), (255, 0, 255), thickness=thickness)

        ##top right
        cv2.line(img, (x1, y), (x1-length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y + length), (255, 0, 255), thickness=thickness)

        ##buttom left
        cv2.line(img, (x, y1), (x+length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - length), (255, 0, 255), thickness=thickness)

        ## buttom right
        cv2.line(img, (x1, y1), (x1-length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - length), (255, 0, 255), thickness=thickness)


    

    

def main():
    cap = cv2.VideoCapture('FaceVideos/3.mp4')
    pTime = 0
    detector = FaceDetector(model_selection=0)

    while True:
        _, img = cap.read()
        bboxs = detector.findFaces(img) 
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 2)
        #showing image
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()