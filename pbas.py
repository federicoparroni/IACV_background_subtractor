import numpy as np
import cv2
import random

class PBAS():

    # N:
    # K:
    # ...
    def __init__(self, N, K, R_incdec, R_lower, R_scale, T_dec, T_inc, T_lower, T_upper):
        self.N = N
        self.K = K
        self.R_incdec = R_incdec
        self.R_lower = R_lower
        self.R_scale = R_scale
        self.T_dec = T_dec
        self.T_inc = T_inc
        self.T_lower = T_lower
        self.T_upper = T_upper

        self.frame_shape = None
        self.current_frame_index = 1
        self.B = None
        self.R = None
        self.T = None
        self.F = None

    def _distance(self, a, b):
        return abs(a-b)

    # Build the segmentation mask F
    def _segment(self, frame):

        B_copy = self.B.copy()
        R_copy = self.R.copy()

        # a pixel (x,y) is foreground (so F(x,y)=1) if the distance between (x,y) and at least K
        # of the N background values is less than R(x,y)
        for x in range(self.frame_shape[0]):
            for y in range(self.frame_shape[1]):
                k = [0, 0, 0]
                c = 0
                while c < 3 or k[c] >= self.K:
                    j = 0
                    while j < min(self.N, self.current_frame_index) or k[c] >= self.K:
                        if self._distance(frame[x,y,c], B_copy[j,x,y,c]) < R_copy(x,y,c):
                            k[c] += 1
                        j += 1
                    c += 1
                # check if at least K distances are less than R(x,y)
                if k[c] >= self.K:
                    self.F[x,y] = 1
                else:
                    self.F[x, y] = 0
                    self._bgupdate(frame, x, y)

    def _bgupdate(self, frame, x, y):
        #calculate if an update is performed p = 1/t
        if random.uniform(0, 1) > self.T[x, y]:
            #choose one of the N frames to update
            n = random.randint(0, 30)
            #update the pixel of the choosen B with the one of the actual frame
            self.B[n, x, y] = frame[x, y]
            #TODO: WE HAVE CHOOSEN TO UPDATE ALSO A NEIGHBOUR PIXEL EVERY TIME WE UPDATE THE FIRST ONE

            d = [-1, 1, 0]

            y_disp = 0
            x_disp = 0
            while (x_disp == 0 and y_disp == 0) or x+x_disp > self.frame_shape(0) or y+y_disp > self.frame_shape(1):
                y_disp = d[random.randint(0, 3)]
                x_disp = d[random.randint(0, 3)]
            self.B[n, x+x_disp, y+y_disp] = frame[x+x_disp, y+y_disp]

            #call the updateR
            self._updateR(frame, n, x, y)



    def _updateR(self, frame):
        pass

    def _updateT(self, frame):
        pass


    def process(self, frame):
        if self.frame_shape is None:
            self.frame_shape = frame.shape

        #insert the N as first shape dimension.
        shape = np.insert(self.frame_shape, 0, self.N)

        if self.B is None:
            #shape structure B: [N, Y_pixel, X_pixel, 1]
            self.B = np.zeros(shape=shape, dtype=np.uint8)

        if self.R is None:
            # shape structure R: [Y_pixel, X_pixel, 1]
            self.R = np.zeros(self.frame_shape, np.float)

        if self.T is None:
            # shape structure T: [Y_pixel, X_pixel, 1]
            self.T = np.zeros(self.frame_shape, np.float)

        if self.F is None:
            # shape structure fg_mask: [Y_pixel, X_pixel]
            shape_fg_mask = np.delete(self.frame_shape, -1)
            self.F = np.zeros(shape_fg_mask, np.uint8)


        self._segment(frame)
        self._bgupdate(frame)
        self._updateR(frame)
        self._updateT(frame)



        self.current_frame_index += 1
        return self.F


