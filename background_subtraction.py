from background_estimation import background_estimate
import cv2
import numpy as np
import time

video = 'videos/svincolo.mp4'
bg = background_estimate(video, False)

cap = cv2.VideoCapture(video)
threshold = 20
while True:
    ret,frame = cap.read()
    if ret == False:
        break
    
    mask = np.copy(frame)
    mask[np.abs(frame - bg) > threshold] = 0
    mask += 1
    mask = np.product(mask, axis=2)
    mask -= 1
    mask[mask>0]=255
    mask = mask.astype(np.uint8)

    frame = cv2.bitwise_and(frame,frame, mask=mask)

    cv2.imshow('window-name',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    time.sleep(0.02)

cap.release()
cv2.destroyAllWindows()
