import cv2
import numpy as np
import time

def read_frame():
    video = 'videos/svincolo.mp4'
    cap = cv2.VideoCapture(video)

    kernel_e = np.ones((3,3), np.uint8)

    while True:
        ret,frame = cap.read()
        print(frame.shape)
        if ret == False:
            break
        
        cv2.imshow('window-name',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        time.sleep(0.02)

    cap.release()
    cv2.destroyAllWindows()

read_frame()