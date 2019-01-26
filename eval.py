import numpy as np
import cv2 as cv
import os
import time
from MOG import MOG
from tqdm import tqdm

def demo(bp, algo, grey=False):
    images = sorted(os.listdir(bp + 'input/'))
    print('demo on ' + bp)
    for i in tqdm(range(len(images))):
        j = images[i]
        image = cv.imread(bp + 'input/' + j)
        if grey:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        mask = algo.process(image)
        mask = np.logical_not(mask).astype(np.uint8)
        image = cv.bitwise_and(image, image, mask=mask)
        cv.imshow('frame', image)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

def evaluate(l, algo, grey=False):
    for bp in l:
        images = sorted(os.listdir(bp + 'input/'))
        gts = sorted(os.listdir(bp + 'groundtruth/'))
        print('evaluating on ' + bp)
        temporal_ROI = [int(x) for x in open(bp + 'temporalROI.txt').read().split(' ')]
        tp = 0; fp = 0; tn = 0; fn = 0

        for i in tqdm(range(len(images))):
            j = images[i]
            image = cv.imread(bp + 'input/' + j)
            if grey:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            mask = algo.process(image)

            if i+1 >= temporal_ROI[0] and i+1 <= temporal_ROI[1]:
                k = gts[i]
                gt = cv.imread(bp + 'groundtruth/' + k)
                gt = np.all(gt, axis=2).astype(np.uint8)

                tp += np.sum(np.logical_and(gt,mask))
                tn += np.sum(np.logical_not(np.logical_or(gt,mask)))
                fn += np.sum(np.logical_and(gt, np.logical_xor(gt,mask)))
                fp += np.sum(np.logical_and(mask, np.logical_xor(gt,mask)).astype(np.uint8))

                # cv.imshow('Frame',mask)
                # if cv.waitKey(25) & 0xFF == ord('q'):
                #     print('eheh')
        re = tp/(tp+fn)
        pr = tp/(tp+fp)
        f1 = (2*pr*re)/(pr+re)
        sp = tn/(tn+fp)
        fpr = fp/(fp+tn)
        fnr = fn/(tn+fp)
        pwc = 100*(fn+fp)/(tp+fn+fp+tn)
        print('results: tp = {}, tn = {}, fp = {}, fn = {}. (avg) recall = {}, specificity = {}, false positive rate = {}, false negative rate = {}, percentage of wrong classification = {}, precision = {}, f1 score = {}'.format(tp, tn, fp, fn, re, sp, fpr, fnr, pwc, pr, f1))

demo('dataset/dataset/baseline/highway/', MOG(), grey=True)
# evaluate(['dataset/dataset/baseline/highway/'], m)