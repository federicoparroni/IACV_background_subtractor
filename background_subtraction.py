from background_estimation import background_estimate
import cv2
import numpy as np
import time
from scipy.signal import convolve2d
import scipy.ndimage

video = 'videos/svincolo.mp4'
bg = background_estimate(video, False)

cap = cv2.VideoCapture(video)
threshold = 300
kernel_e = np.ones((3,3), np.uint8)
# kernel_d = np.ones((5,5), np.uint8)
k = np.ones((5, 5), np.uint8)
old_mask = np.ones(bg.shape[0:2])

while True:
    ret,frame = cap.read()
    if ret == False:
        break
    
    # obtain the mask by background differencing
    mask = np.all(np.power(np.abs(frame - bg), 3) < threshold, axis=2)

    # filter mask based on the number of non masked neighbors
    filter = convolve2d(mask, k)[2:-2, 2:-2]
    mask[filter > 1] = 1

    # filter mask based on the number of unchanged neighbors
    # filter = np.logical_xor(mask, old_mask)
    # old_mask = np.copy(mask)
    # filter = convolve2d(filter, k)[1:-1, 1:-1]
    # mask[filter > 5] = 1

    # morphological op on mask
    mask = np.logical_not(scipy.ndimage.binary_fill_holes(np.logical_not(mask).astype(np.uint8))).astype(np.uint8)
    # mask = cv2.dilate(mask, kernel_d, iterations=1)
    mask = cv2.erode(mask, kernel_e, iterations=1) 

    mask = mask.astype(np.uint8)

    # mask the frame
    frame = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('window-name',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    time.sleep(0.02)

cap.release()
cv2.destroyAllWindows()
