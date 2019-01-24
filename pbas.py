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

        self.current_frame_index = 0
        self.B = None

    def _segment(self, frame):
        pass

    def _bgupdate(self, frame):
        pass

    def _updateR(self, frame):
        pass

    def _updateT(self, frame):
        pass


    def process(self, frame):
        if self.B is None:
            self.B = np.zeros(frame.shape, np.uint8)

        fg_mask = self._segment(frame)
        self._bgupdate(frame)
        self._updateR(frame)
        self._updateT(frame)

        self.current_frame_index += 1
        return fg_mask
