import cv2
import time
from pbas_algorithm import PBAS_algorithm

def read_frame():
    video = 'videos/salitona.mp4'
    cap = cv2.VideoCapture(video)
    p = PBAS_algorithm()

    while True:
        start = time.time()

        ret, frame = cap.read()
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = p.process(frame_grayscale)
        if ret == False:
            break

        print(time.time()-start)

        cv2.imshow('window-name', frame_grayscale)
        cv2.imshow('window-name', mask)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        time.sleep(0.02)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    read_frame()
