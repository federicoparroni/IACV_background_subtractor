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
        self._fg_mask = None

    def _segment(self, frame):
        if K > self.current_frame_index:
            return self._fg_mask

        for x in range(self.frame_shape[0]):
            for y in range(self.frame_shape[1]):
                for c in range(3):
                    j = 0
                    k = 0
                    while j < min(N, self.current_frame_index) or k >= K:
                        if self._distance() < 

    def _bgupdate(self, frame):
        pass

    def _updateR(self, frame):
        pass

    def _updateT(self, frame):
        pass


    def process(self, frame):
        if self.frame_shape is None:
            self.frame_shape = frame.shape
        if self.B is None:
            self.B = np.zeros(frame.shape, np.uint8)
        if self.R is None:
            self.R = np.zeros(frame.shape, np.float)
        if self.T is None:
            self.T = np.zeros(frame.shape, np.float)
        if self.fg_mask is None:
            self.fg_mask = np.zeros(frame.shape, np.uint8)

        self._fg_mask = self._segment(frame)
        self._bgupdate(frame)
        self._updateR(frame)
        self._updateT(frame)

        self.current_frame_index += 1
        return self._fg_mask
