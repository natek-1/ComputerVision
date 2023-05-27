import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture('FaceVideos/7.mp4')
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


pTime = 0
while True:
    _, img = cap.read()
    #Getting the frame rate of the video
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            # using our own drawing and information to make a solid drawing of the face in the image
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h) , int(bboxC.width *w), int(bboxC.height*h)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f"Certainty: {int(detection.score[0]*100)}%",
                         (bbox[0],bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 2)
    #showing image
    cv2.imshow("Image", img)
    cv2.waitKey(1)