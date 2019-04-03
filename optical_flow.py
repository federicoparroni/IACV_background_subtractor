import cv2
import numpy as np

FLOW_MAG_THRESHOLD = 0.7
CONSECUTIVE_FLOW_FRAMES = 6
# FLOW_ANGLE_THRESHOLD = 0.3

cap = cv2.VideoCapture('dataset/Jackson_Hole_Wyoming/out0.mov')
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
hsv[...,2] = 255

# avg_flow_angle = np.zeros(frame1.shape[:2], dtype=float)
# avg_mag = np.zeros_like(avg_flow_angle)
scores = np.zeros(frame1.shape[:2], dtype=np.uint8)
flow_mag_history = []

def update_flow(mag, scores):
    indices = mag > FLOW_MAG_THRESHOLD
    hot = np.zeros(mag.shape, dtype=np.uint8)
    hot[indices] = 255

    flow_mag_history.append(hot)

    if len(flow_mag_history) >= CONSECUTIVE_FLOW_FRAMES:
        flow_mag_history.pop(0)

        total_hot = np.ones(mag.shape, dtype=np.uint8) * 255
        for m in flow_mag_history:
            total_hot = cv2.bitwise_and(total_hot, m)
        scores = cv2.bitwise_or(scores, total_hot)
        return scores
    else:
        return np.zeros(mag.shape, dtype=np.uint8)

n = 1
while(1):
    ret, frame2 = cap.read()
    cv2.imshow('frame', frame2)
    
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # threshold
    # indices = mag > FLOW_MAG_THRESHOLD
    # hot = np.zeros(mag.shape, dtype=np.uint8)
    # hot[indices] = 255
    # cv2.imshow('mag', hot)

    scores = update_flow(mag, scores)
    cv2.imshow('mag', scores)

    # show optical flow
    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,1] = 0
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    # cv2.imshow('flow',bgr)

    if cv2.waitKey(25) & 0xff == ord('q'): break
    prvs = next
    n += 1

cap.release()
cv2.destroyAllWindows()


# update average flow direction
    # def flow_with_angle():
    #     ang[ang >= np.pi] -= np.pi
    #     flow_indices = mag > FLOW_MAG_THRESHOLD
    #     avg_flow_angle[flow_indices] += (ang[flow_indices] - avg_flow_angle[flow_indices]) / n
    #     differences = np.abs(avg_flow_angle - ang)
    #     scores[(differences < FLOW_ANGLE_THRESHOLD) & (flow_indices) & (scores < 255)] += 1
    #     cv2.imshow('scores', scores)