import numpy as np
import random
import time
from tqdm import tqdm

class PBAS_algorithm:

    # N:
    # K:
    # ...
    def __init__(self, N=30, K=2, R_incdec=0.05, R_lower=18, R_scale=5, T_dec=0.05, T_inc=1, T_lower=2, T_upper=200):
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
        self.D = None
        self.R = None
        self.T = None
        self.F = None
        self.d_minavg = None

    def _distance(self, a, b):
        return abs(a-b)

    # Build the segmentation mask F
    def _segment(self, frame):

        B_copy = self.B.copy()
        R_copy = self.R.copy()

        # a pixel (x,y) is foreground (so F[x,y]=1) if the distance between (x,y) and at least K
        # of the N background values is less than R[x,y]
        for x in range(self.frame_shape[0]):
            for y in range(self.frame_shape[1]):

                #c = 0
                #while c < 3 or k >= self.K:
                k = 0       # number of lower-than-R distances for the channel 'c'
                j = 0
                frame_pixel = frame[x,y]
                r = R_copy[x,y]
                while j < self.N or k >= self.K:
                    if self._distance(frame_pixel, B_copy[j,x,y]) < r:
                        k += 1
                    j += 1
                # check if at least K distances are less than R(x,y)
                if k >= self.K:
                    self.F[x,y] = 1
                else:
                    self.F[x, y] = 0
                    self._bgupdate(frame, x, y)


    def _bgupdate(self, frame, x, y):
        #calculate if an update is performed p = 1/t
        if random.uniform(0, 1) > self.T[x, y]:
            #choose one of the N frames to update
            n = random.randint(0, 29)
            #update the pixel of the choosen B with the one of the actual frame
            self.B[n, x, y] = frame[x, y]
            #TODO: WE HAVE CHOOSEN TO UPDATE ALSO A NEIGHBOUR PIXEL EVERY TIME WE UPDATE THE FIRST ONE

            d = [(-1, -1), (-1, 1), (-1, 0), (1, -1), (1, 1), (1, 0), (0, 1), (0, -1)]

            y_disp = 0
            x_disp = 0

            while (x_disp == 0 and y_disp == 0) or x+x_disp >= self.frame_shape[0] or y+y_disp >= self.frame_shape[1]:
                x_disp = d[random.randint(0, 7)][0]
                y_disp = d[random.randint(0, 7)][1]

            self.B[n, x+x_disp, y+y_disp] = frame[x+x_disp, y+y_disp]




            #call the updateR
            self._updateR(frame, n, x, y)
            self._updateR(frame, n, x+x_disp, y+y_disp)



    def _updateR(self, frame, n, x, y):
        self.D[n, x, y] = min([self._distance(frame[x, y], self.B[n, x, y]) for n in range(self.N)])
        self.d_minavg[x, y] = np.mean(self.D[:, x, y])
        if self.R[x, y] > self.d_minavg[x, y]*self.R_scale:
            self.R[x, y] = self.R[x, y]*(1-self.R_incdec)
        else:
            self.R[x, y] = self.R[x, y]*(1+self.R_incdec)

    def _updateT(self):
        for x in range(self.frame_shape[0]):
            for y in range(self.frame_shape[1]):
                #for c in range(3):
                Tinc_over_dmin = self.T_inc / self.d_minavg[x, y]
                if self.F[x, y] == 1:
                    self.T[x, y] += Tinc_over_dmin
                else:
                    self.T[x, y] -= Tinc_over_dmin
                self.T[x, y] = max(self.T_lower, self.T[x, y])
                self.T[x, y] = min(self.T[x, y], self.T_upper)

    def process(self, frame):

        if self.frame_shape is None:
            self.frame_shape = frame.shape

        # insert the N as first shape dimension.
        shape = np.insert(self.frame_shape, 0, self.N)

        if self.B is None:
            #shape structure B: [N, X_pixel, Y_pixel, 1]
            self.B = np.zeros(shape=shape, dtype=np.uint8)

        if self.D is None:
            # shape structure B: [N, X_pixel, Y_pixel, 3]
            self.D = np.ones(shape=shape)*np.inf

        if self.R is None:
            # shape structure R: [X_pixel, Y_pixel, 1]
            self.R = np.zeros(self.frame_shape, np.float)

        if self.d_minavg is None:
            self.d_minavg = np.zeros(self.frame_shape, np.float)

        if self.T is None:
            # shape structure T: [X_pixel, Y_pixel, 1]
            self.T = np.zeros(self.frame_shape, np.float)

        if self.F is None:
            # shape structure fg_mask: [X_pixel, Y_pixel]
            shape_fg_mask = [self.frame_shape[0], self.frame_shape[1]]
            self.F = np.zeros(shape_fg_mask, np.uint8)

        #start = time.time()
        self._segment(frame)
        #print(time.time() - start)
        self._updateT()


        self.current_frame_index += 1
        return self.F




