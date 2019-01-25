import numpy as np
import cv2

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
        self.current_frame_index = 0
        self.B = None
        self.R = None
        self.T = None
        self.F = None
        self.d_minavg = None

    def _distance(self, a, b):
        return abs(a-b)

    # Build the segmentation mask F
    def _segment(self, frame):
        # a pixel (x,y) is foreground (so F[x,y]=1) if the distance between (x,y) and at least K
        # of the N background values is less than R[x,y]
        for x in range(self.frame_shape[0]):
            for y in range(self.frame_shape[1]):
                #c = 0
                #while c < 3 or k >= self.K:
                k = 0       # number of lower-than-R distances for the channel 'c'
                j = 0
                while j < self.N or k >= self.K:
                    if self._distance(frame[x,y], self.B[j,x,y]) < self.R[x,y]:
                        k += 1
                    j += 1
                # check if at least K distances are less than R(x,y)
                if k >= self.K:
                    self.F[x,y] = 1
                else:
                    self.F[x,y] = 0
                    self._bgupdate(frame, x,y)
                #c += 1

    def _bgupdate(self, frame):
        pass

    def _updateR(self, frame):
        pass

    def _updateT(self, frame):
        for x in range(self.frame_shape[0]):
            for y in range(self.frame_shape[1]):
                #for c in range(3):
                Tinc_over_dmin = self.T_inc / self.d_minavg[x,y]
                if self.F[x,y] == 1:
                    self.T[x,y] += Tinc_over_dmin
                else:
                    self.T[x,y] -= Tinc_over_dmin
                self.T[x,y] = max(self.T_lower, self.T[x,y])
                self.T[x,y] = min(self.T[x,y], self.T_upper)


    def process(self, frame):
        if self.frame_shape is None:
            self.frame_shape = frame.shape

        #insert the N as first shape dimension.
        shape = np.insert(self.frame_shape, 0, self.N)

        if self.B is None:
            #shape structure B: [N, Y_pixel, X_pixel, 3]
            self.B = np.zeros(shape=shape, dtype=np.uint8)

        if self.R is None:
            # shape structure R: [Y_pixel, X_pixel, 3]
            self.R = np.zeros(self.frame_shape, np.float)

        if self.T is None:
            # shape structure T: [Y_pixel, X_pixel, 3]
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
