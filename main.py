#!/usr/bin/env python

'''
Video object detection.

Usage:
    main.py [-h] [--detect_class DETECT_CLASS] [--voice_cmd]
               net_model detect_mode

    net_model:
        0 - MobileNet SSD V1 COCO
        1 - MobileNet SSD V1 BALLS

    detect_mode:
        1 - detect all objects
        2 - detect a specific object
        3 - track a specific object

    --detect_class:
        Required when detection mode > 1

    --voice_cmd:
        Enable voice commands

Keys:
    ENTER  - change detected object
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from numpy import pi, sin, cos

import cv2 as cv
import threading
import speech_recognition as sr
import pyaudio
import wave

cvNet = None
myName = 'computer'
showVideoStream = False
currentClassDetecting = 'background'
audio_yes = 'audio/yes.wav'
audio_okay = 'audio/okay.wav'
audio_invalid = 'audio/invalid.wav'

netModels = [
    {
        'modelPath': 'models/mobilenet_ssd_v1_coco/frozen_inference_graph.pb',
        'configPath': 'models/mobilenet_ssd_v1_coco/ssd_mobilenet_v1_coco_2017_11_17.pbtxt',
        'classNames': { 
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
    },
    {
        'modelPath': 'models/mobilenet_ssd_v1_balls/transformed_frozen_inference_graph.pb',
        'configPath': 'models/mobilenet_ssd_v1_balls/ssd_mobilenet_v1_balls_2018_05_20.pbtxt',
        'classNames': {
            0: 'background', 1: 'red ball'
        }
    }
]


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

def label_class(img, detection, score, className, boxColor=None):
    rows = img.shape[0]
    cols = img.shape[1]

    if boxColor == None:
        boxColor = (23, 230, 210)
    
    xLeft = int(detection[3] * cols)
    yTop = int(detection[4] * rows)
    xRight = int(detection[5] * cols)
    yBottom = int(detection[6] * rows)
    cv.rectangle(img, (xLeft, yTop), (xRight, yBottom), boxColor, thickness=4)

    label = className + ": " + str(int(round(score * 100))) + '%'
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    yTop = max(yTop, labelSize[1])
    cv.rectangle(img, (xLeft, yTop - labelSize[1]), (xLeft + labelSize[0], yTop + baseLine),
        (255, 255, 255), cv.FILLED)
    cv.putText(img, label, (xLeft, yTop), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    pass

def detect_all_objects(img, detections, score_threshold, classNames):
    for detection in detections:
        class_id = int(detection[1])
        score = float(detection[2])
        if score > score_threshold:
            label_class(img, detection, score, classNames[class_id])
    pass

def detect_object(img, detections, score_threshold, classNames, className):
    for detection in detections:
        score = float(detection[2])
        class_id = int(detection[1])
        if className in classNames.values() and className == classNames[class_id] and score > score_threshold:
            label_class(img, detection, score, classNames[class_id])
    pass

def track_object(img, detections, score_threshold, classNames, className, tracking_threshold):
    for detection in detections:
        score = float(detection[2])
        class_id = int(detection[1])
        if className in classNames.values() and className == classNames[class_id] and score > score_threshold:
            rows = img.shape[0]
            cols = img.shape[1]
            marginLeft = int(detection[3] * cols) # xLeft
            marginRight = cols - int(detection[5] * cols) # cols - xRight
            xMarginDiff = abs(marginLeft - marginRight)
            marginTop = int(detection[4] * rows) # yTop
            marginBottom = rows - int(detection[6] * rows) # rows - yBottom
            yMarginDiff = abs(marginTop - marginBottom)
            
            if xMarginDiff < tracking_threshold and yMarginDiff < tracking_threshold:
                boxColor = (0, 255, 0)
            else:
                boxColor = (0, 0, 255)

            label_class(img, detection, score, classNames[class_id], boxColor)
    pass

def play_audio(audioFile):
    chunk = 1024
    wf = wave.open(audioFile, 'rb')
    pa = pyaudio.PyAudio()
    stream = pa.open(format = pa.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)

    data = wf.readframes(chunk)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    pa.terminate()

def run_voice_command(classNames):
    rc = sr.Recognizer()
    while showVideoStream:
        mic = sr.Microphone()
        with mic as source:
            rc.adjust_for_ambient_noise(source)
            try:
                audio = rc.listen(source)
                result = rc.recognize_google(audio).lower()
                if result == myName:
                    play_audio(audio_yes)
                    print('Say command')
                    audio = rc.listen(source)

                    result = rc.recognize_google(audio).lower()
                    if result in classNames.values():
                        global currentClassDetecting
                        currentClassDetecting = result
                        print('Now detecting: ' + result)
                        play_audio(audio_okay)
                    else:
                        print('The object ' + result + ' is invalid')
                        play_audio(audio_invalid)

            except:
                pass # ignore unrecognizable audio

    print('exiting run_voice_command...')
    pass


def run_video_detection(mode, netModel):
    scoreThreshold = 0.3
    trackingThreshold = 50

    cvNet = cv.dnn.readNetFromTensorflow(netModel['modelPath'], netModel['configPath'])
    cap = create_capture()
    
    global showVideoStream
    while showVideoStream:
        ret, img = cap.read()

        # run detection
        cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        detections = cvNet.forward()

        if mode == 1:
            detect_all_objects(img, detections[0,0,:,:], scoreThreshold, netModel['classNames'])
        elif mode == 2:
            detect_object(img, detections[0,0,:,:], scoreThreshold, netModel['classNames'], currentClassDetecting)
        elif mode == 3:
            track_object(img, detections[0,0,:,:], scoreThreshold, netModel['classNames'], currentClassDetecting, trackingThreshold)
        
        cv.imshow('Real-Time Object Detection', img)

        ch = cv.waitKey(1)
        if ch == 27:
            showVideoStream = False
            break

    print('exiting run_video_detection...')
    cv.destroyAllWindows()
    pass

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net_model", type=int, help="The network model id: \
        0 - MobileNet SSD V1 COCO \
        1 - MobileNet SSD V1 BALLS")
    parser.add_argument("detect_mode", type=int, help="The detection mode: \
        1 - detect all objects \
        2 - detect a specific object \
        3 - track a specific object")
    parser.add_argument("--detect_class", help="The class to detect. Required when mode > 1")
    parser.add_argument("--voice_cmd", help="Enable voice commands", action="store_true")
    args = parser.parse_args()
    
    if args.detect_mode > 1:
        if args.detect_class is None:
            print("Error: You must specify a class to detect if detection mode > 1")
            sys.exit(0)
        else:
            currentClassDetecting = args.detect_class
    
    showVideoStream = True
    videoStreamThread = threading.Thread(target=run_video_detection, args=[args.detect_mode,netModels[args.net_model]])
    videoStreamThread.start()

    if args.voice_cmd:
        voiceCommandThread = threading.Thread(target=run_voice_command, args=[netModels[args.net_model]['classNames']])
        voiceCommandThread.start()
