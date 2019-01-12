from background_estimation import background_estimate
import cv2
import numpy as np
import time

video = 'videos/traffic.mp4'
bg = background_estimate(video, False)

cap = cv2.VideoCapture(video)
thr = 5
while True:
    ret,frame = cap.read()
    if ret == False:
        break

    mask = np.copy(frame, )
    mask[np.abs(frame - bg) > 10] = 0
    mask = np.product(mask, axis=2)
    mask[mask>0]=255
    mask = mask.astype(np.uint8)

    frame = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('window-name',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    time.sleep(0.02)

cap.release()
cv2.destroyAllWindows()
