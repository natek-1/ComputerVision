import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.complexity = model_complexity
        self.landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = min_detection_confidence
        self.tracking_confidence = min_tracking_confidence
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                      model_complexity=self.complexity,
                                      smooth_landmarks=self.landmarks,
                                      enable_segmentation=self.enable_segmentation,
                                      smooth_segmentation=self.smooth_segmentation,
                                      min_detection_confidence=self.detection_confidence,
                                      min_tracking_confidence=self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.results= None

    
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                    self.mpPose.POSE_CONNECTIONS)
    
    def getPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    # Note: lm.x and lm.y is not going to be the position of points on the image, rather it give the point ration 
                    # I muliply by width and height to get the pixel coordinate
                h, w, _= img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255, 0,0), cv2.FILLED)
        return lmList
        




def main():
    cap = cv2.VideoCapture('PoseVideos/3.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        detector.findPose(img)
        lmList = detector.getPosition(img)
        # trying to find the fps of the video that is playing
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        # Adding text so frame rate is visible
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0,0), 3)

        cv2.imshow("Image:", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()