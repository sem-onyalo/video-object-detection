#!/usr/bin/env python

'''
Video object detection.

Usage:
    main.py <detection mode> <class name>

    detection mode:
        1 - detect all objects
        2 - detect a specific object

Keys:
    ESC    - exit

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from numpy import pi, sin, cos

import cv2 as cv

cvNet = None

classNames = { 
    0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
    80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' 
}


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

def label_class(img, detection, score, class_id):
    rows = img.shape[0]
    cols = img.shape[1]
    
    left = int(detection[3] * cols)
    top = int(detection[4] * rows)
    right = int(detection[5] * cols)
    bottom = int(detection[6] * rows)
    cv.rectangle(img, (left, top), (right, bottom), (23, 230, 210), thickness=2)

    
    label = classNames[class_id] + ": " + str(score)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(img, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine),
        (255, 255, 255), cv.FILLED)
    cv.putText(img, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    pass

def detect_all_objects(img, detections, score_threshold):
    for detection in detections:
        class_id = int(detection[1])
        score = float(detection[2])
        if score > score_threshold:
            label_class(img, detection, score, class_id)
    pass

def detect_object(img, detections, score_threshold, className):
    for detection in detections:
        score = float(detection[2])
        class_id = int(detection[1])
        if className in classNames.values() and className == classNames[class_id] and score > score_threshold:
            label_class(img, detection, score, class_id)
    pass

if __name__ == '__main__':
    import sys
    import getopt

    print(__doc__)
    
    sources = [ 0 ] # use default video source (webcam)
    scoreThreshold = 0.3
    
    args = sys.argv[1:]
    mode = int(args[0])
    
    cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v1_coco_2017_11_17.pbtxt')
    caps = list(map(create_capture, sources))
    
    while True:
        for i, cap in enumerate(caps):
            ret, img = cap.read()

            # run detection
            cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
            detections = cvNet.forward()

            if mode == 1:
                detect_all_objects(img, detections[0,0,:,:], scoreThreshold)
            elif mode == 2:
                className = args[1]
                detect_object(img, detections[0,0,:,:], scoreThreshold, className)
            
            cv.imshow('capture %d' % i, img)
        ch = cv.waitKey(1)
        if ch == 27:
            break
    cv.destroyAllWindows()
