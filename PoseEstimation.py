import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose

pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture('PoseVideos/4.mp4')
pTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            # Note: lm.x and lm.y is not going to be the position of points on the image, rather it give the point ration 
            # I muliply by width and height to get the pixel coordinate
            h, w, c= img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx,cy), 10, (255, 0,0), cv2.FILLED)


    # trying to find the fps of the video that is playing
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # Adding text so frame rate is visible
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0,0), 3)

    cv2.imshow("Image:", img)
    cv2.waitKey(1)
