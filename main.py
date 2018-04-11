#!/usr/bin/env python

'''
Video object detection.

Keys:
    ESC    - exit

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from numpy import pi, sin, cos

import cv2 as cv

# built-in modules
from time import clock

cvNet = None

class VideoSynthBase(object):
    def __init__(self, size=None, noise=0.0, bg = None, **params):
        self.bg = None
        self.frame_size = (640, 480)
        if bg is not None:
            self.bg = cv.imread(bg, 1)
            h, w = self.bg.shape[:2]
            self.frame_size = (w, h)

        if size is not None:
            w, h = map(int, size.split('x'))
            self.frame_size = (w, h)
            self.bg = cv.resize(self.bg, self.frame_size)

        self.noise = float(noise)

    def render(self, dst):
        pass

    def read(self, dst=None):
        w, h = self.frame_size

        if self.bg is None:
            buf = np.zeros((h, w, 3), np.uint8)
        else:
            buf = self.bg.copy()

        self.render(buf)

        if self.noise > 0.0:
            noise = np.zeros((h, w, 3), np.int8)
            cv.randn(noise, np.zeros(3), np.ones(3)*255*self.noise)
            buf = cv.add(buf, noise, dtype=cv.CV_8UC3)
        return True, buf

    def isOpened(self):
        return True

def create_capture(source = 0):
    source = str(source).strip()
    chunks = source.split(':')
    # handle drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]
    try: source = int(source)
    except ValueError: pass
    params = dict( s.split('=') for s in chunks[1:] )
    
    cap = cv.VideoCapture(source)
    if 'size' in params:
        w, h = map(int, params['size'].split('x'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
    return cap

def run_detection(img):
    rows = img.shape[0]
    cols = img.shape[1]
    cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
    return img

if __name__ == '__main__':
    import sys
    import getopt

    print(__doc__)

    cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v1_coco_2017_11_17.pbtxt')
    sources = [ 0 ] # use default video source (webcam)
    caps = list(map(create_capture, sources))
    shot_idx = 0
    while True:
        imgs = []
        for i, cap in enumerate(caps):
            ret, img = cap.read()
            img = run_detection(img)
            imgs.append(img)
            cv.imshow('capture %d' % i, img)
        ch = cv.waitKey(1)
        if ch == 27:
            break
    cv.destroyAllWindows()
