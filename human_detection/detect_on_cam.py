import torch
import numpy as np
import cv2 as cv

import time

from human_detect import human_detection

human_detection = human_detection()

def video_demo():
    t = time.time()
    # 0是代表摄像头编号，只有一个的话默认为0
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while (True):
        t_old = t
        t = time.time()
        fps = 1/(t-t_old)

        ref, frame = cap.read()

        human = human_detection.human_detect(frame)
        cv.imshow("1", frame)

        # 等待30ms显示图像，若过程中按“Esc”退出
        c = cv.waitKey(30) & 0xff
        if c == 27:
            cap.release()
            break


video_demo()
cv.destroyAllWindows()
