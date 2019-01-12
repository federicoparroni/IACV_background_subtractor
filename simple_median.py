import numpy as np
import matplotlib.pyplot as plt
import cv2

import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('Qt4Agg')

cap = cv2.VideoCapture('videos/salitona.mp4')
V = []
while True:
    ret,frame = cap.read()
    if ret == False:
        break
    V.append(frame)
    cv2.imshow('window-name',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

V = np.array(V)
bg = np.zeros((V[0].shape[0], V[0].shape[1], 3), dtype=np.float32)
bg[:,:,0] = np.median(V[:,:,:,0], axis=0)/255
bg[:,:,1] = np.median(V[:,:,:,1], axis=0)/255
bg[:,:,2] = np.median(V[:,:,:,2], axis=0)/255

plt.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
plt.show()
