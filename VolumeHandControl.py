import cv2
import time
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm
import math
import osascript
from subprocess import call
import subprocess

## Camera
cam_width, cam_height = 640, 480
THUMB_TIP_POINT = 4
INDEX_FINGER_TIP = 8
MIN_DISTANCE = 20
MAX_DISTANCE = 220
MIN_VOLUME = 0
MAX_VOLUME = 100


cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)
success = True
pTime = 0
detector  = htm.HandDetector(maxHands=2,detectionConfidence=0.8)

def find_volume() -> int:
    result = osascript.osascript('get volume settings')
    volInfo = result[1].split(',')
    outputVol = int(volInfo[0].replace('output volume:', ''))
    return outputVol


while success:
    success, img = cap.read()
    detector.findHands(img=img, draw=False)
    lm_list = detector.findPosition(img, draw=False)
    if len(lm_list) != 0:
        # Getting the position of both thump tip and index finger tip from list of landmarks
        x1, y1 = lm_list[THUMB_TIP_POINT][1], lm_list[THUMB_TIP_POINT][2]
        x2, y2 = lm_list[INDEX_FINGER_TIP][1], lm_list[INDEX_FINGER_TIP][2]
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        length = math.hypot(x2 - x1, y2 - y1)
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        ## from testing hand range is from 15 to 280
        ## Volume range is from 0 100
        target_volume = np.interp(length, [MIN_DISTANCE, MAX_DISTANCE], [MIN_VOLUME, MAX_VOLUME])
        vol = f"osascript -e 'set volume output volume {str(target_volume)}'"
        call([vol], shell=True)
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    current_volume = find_volume() 
    cv2.putText(img, f"Volume: {current_volume}%", (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1) # display volume level
    current_volume = np.interp(current_volume, [0, 100], [400, 150])
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3) # Outline for the volume bar
    cv2.rectangle(img, (50, int(current_volume)), (85, 400), (0, 255, 0), cv2.FILLED) # actual volume being displayed
    

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20,50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
    cv2.imshow("Img", img)
    cv2.waitKey(1)