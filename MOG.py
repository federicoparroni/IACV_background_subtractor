import numpy as np
import cv2 as cv
import os
import time
from tqdm import tqdm

class MOG():

    def __init__(self):
        self.fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

    def process(self, frame):
        fgmask = self.fgbg.apply(frame)
        return fgmask
