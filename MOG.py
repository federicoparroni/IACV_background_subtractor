import numpy as np
import cv2 as cv
from scipy.signal import convolve2d

cap = cv.VideoCapture('videos/svincolo.mp4')
# fgbg = cv.createBackgroundSubtractorMOG2()
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
width = int(cap.get(3))
height = int(cap.get(4))
old_mask = np.ones((height, width))
k = np.ones((3,3), np.uint8)

while(1):
    ret, frame = cap.read()
    if ret == False:
        break
    
    fgmask = fgbg.apply(frame)
    mask = np.logical_not(fgmask).astype(np.uint8)

    mask = mask.astype(np.uint8)
    frame = cv.bitwise_and(frame, frame, mask=mask)

    old_mask = np.copy(mask)

    cv.imshow('frame',frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()