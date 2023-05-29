import mediapipe as mp
import time
import cv2

cap = cv2.VideoCapture("FaceVideos/3.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            #using enumerate to tell us what the id of the landmark is 
            # which can be used to specify which specific landmark you want
            for id, lm in enumerate(faceLms.landmark):
                height, width, _ = img.shape
                x,y = int(lm.x*width), int(lm.y* height)
                


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255, 0), 3)
    if success == False:
        break
    cv2.imshow("Test Image", img)
    cv2.waitKey(1)