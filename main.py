#!/usr/bin/env python

'''
Video object detection.

Usage:
    main.py [-h] [--detect_class DETECT_CLASS] [--voice_cmd]
               net_model detect_mode

    net_model:
        0 - MobileNet SSD V1 COCO
        1 - MobileNet SSD V1 BALLS
        2 - MobileNet SSD V1 BOXING

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
totalFrames = 0
totalObjects = 0
videoWriter = None
doWriteVideo = False
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
            0: 'background', 1: 'red ball', 2: 'blue ball'
        }
    },
    {
        'modelPath': 'models/mobilenet_ssd_v1_boxing/transformed_frozen_inference_graph.pb',
        'configPath': 'models/mobilenet_ssd_v1_boxing/ssd_mobilenet_v1_boxing_2019_02_03.pbtxt',
        'classNames': {
            0: 'background', 1: 'boxing gloves'
        }
    },
    {
        'modelPath': 'models/ssd_inception_v2_coco/frozen_inference_graph.pb',
        'configPath': 'models/ssd_inception_v2_coco/ssd_inception_v2_coco_2017_11_17.pbtxt',
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
        'modelPath': 'models/face_detector/yeephycho_tf_face_detector.pb',
        'classNames': {
            0: 'background', 1: 'face'
        }
    },
    {
        'modelPath': 'models/mobilenet_ssd_v1_mask/frozen_inference_graph.pb',
        'configPath': 'models/mobilenet_ssd_v1_mask/ssd_mobilenet_v1_mask_2021_01_12.pbtxt',
        'classNames': {
            0: 'background', 1: 'mask', 2: 'no mask'
        }
    }
]


def create_capture(source=None):
    if source == None:
        source = 0
    
    cap = cv.VideoCapture(source)
    
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
    return cap

def label_class(img, detection, score, className, boxColor=None, textLabelSettings=None):
    rows = img.shape[0]
    cols = img.shape[1]

    boxThickness = 4
    if boxColor == None:
        boxColor = (23, 230, 210)
    
    xLeft = int(detection[3] * cols)
    yTop = int(detection[4] * rows)
    xRight = int(detection[5] * cols)
    yBottom = int(detection[6] * rows)
    cv.rectangle(img, (xLeft, yTop), (xRight, yBottom), boxColor, boxThickness)

    if textLabelSettings != None:
        text = className
        textColor = textLabelSettings['color']
        textFont = textLabelSettings['font']
        textScale = textLabelSettings['scale']
        textThickness = textLabelSettings['thickness']
        if textLabelSettings['mapping'] != None:
            text = textLabelSettings['mapping']['text'][className]
            textColor = textLabelSettings['mapping']['color'][className]
        boxWidth = int(xRight - xLeft)
        boxHeight = int(yBottom - yTop)
        textSize = cv.getTextSize(text, textFont, textScale, textThickness)
        textWidth = textSize[0][0]
        textHeight = textSize[0][1]

        textX = int(round((boxWidth - textWidth) / 2)) + xLeft
        textY = int(round((boxHeight - textHeight) / 2)) + yTop + textHeight
        
        cv.putText(img, text, (textX, textY), textFont, textScale, textColor, textThickness, cv.LINE_AA)

def detect_all_objects(img, detections, score_threshold, classNames, textLabelSettings=None):
    for detection in detections:
        classId = int(detection[1])
        score = float(detection[2])
        if score > score_threshold:
            label_class(img, detection, score, classNames[classId], textLabelSettings=textLabelSettings)

def detect_object(img, detections, score_threshold, classNames, className, detectAllInstances):
    objCnt = 0
    if detectAllInstances:
        for detection in detections:
            score = float(detection[2])
            classId = int(detection[1])
            if className in classNames.values() and className == classNames[classId] and score > score_threshold:
                objCnt += 1
                label_class(img, detection, score, classNames[classId])
    else: # detect the highest scored instance
        objCnt = 1
        instance = None
        for detection in detections:
            score = float(detection[2])
            classId = int(detection[1])
            if instance == None or (instance != None and instance['score'] < score):
                instance = {
                    'detection': detection,
                    'classId': classId,
                    'score': score
                }

        if instance != None and className in classNames.values() and className == classNames[instance['classId']] and instance['score'] > score_threshold:
            label_class(img, instance['detection'], instance['score'], classNames[instance['classId']])

            rows = img.shape[0]
            cols = img.shape[1]
            xLeft = int(instance['detection'][3] * cols)
            xRight = int(instance['detection'][5] * cols)
            yTop = int(instance['detection'][4] * rows)
            yBottom = int(instance['detection'][6] * rows)
            xFaceTracker = xLeft + int((xRight - xLeft) / 2)
            yFaceTracker = yTop + int((yBottom - yTop) / 3)
            faceTrackerPoint = (xFaceTracker, yFaceTracker)
            faceTrackerPointColor = (0,0,255)
            cv.circle(img, faceTrackerPoint, 4, faceTrackerPointColor, -1)

    global totalObjects
    totalObjects = objCnt

def track_object(img, detections, score_threshold, classNames, className, tracking_threshold):
    for detection in detections:
        score = float(detection[2])
        classId = int(detection[1])
        if className in classNames.values() and className == classNames[classId] and score > score_threshold:
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

            label_class(img, detection, score, classNames[classId], boxColor)

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

def addObjectCountText(img, text, scale=1, thickness=2, widthFactor=1):
    font = cv.FONT_HERSHEY_SIMPLEX
    imgWidth = img.shape[1]
    imgHeight = img.shape[0]

    textSize = cv.getTextSize(text, font, scale, thickness)[0]
    textWidth = textSize[0]
    textHeight = textSize[1]

    textPt = (15, imgHeight - 20)
    rectPt1 = (0, imgHeight - 60)
    rectPt2 = (textWidth + 40, imgHeight)

    cv.rectangle(img, rectPt1, rectPt2, (0, 0, 0), cv.FILLED, cv.LINE_AA)
    cv.putText(img, text, textPt, font, scale, (255, 255, 255), thickness, cv.LINE_AA)

def run_video_detection(input, mode, netModel, scoreThreshold, trackingThreshold, skipFrames, detectAllInstances):
    cvNet = cv.dnn.readNetFromTensorflow(netModel['modelPath'], netModel['configPath'])
    # cvNet = cv.dnn.readTensorFromONNX(netModel['modelPath'])
    cap = create_capture(input)
    
    global showVideoStream
    global doWriteVideo
    global totalFrames
    global videoWriter

    while showVideoStream:
        ret, img = cap.read()

        ch = cv.waitKey(1)
        if ch == 27 or not ret:
            showVideoStream = False
            break

        if totalFrames % skipFrames == 0:
            # run detection
            cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
            detections = cvNet.forward()

            # TODO: pull from config and only apply if model is MASK-NOMASK
            textLabelSettings = {
                'font': cv.FONT_HERSHEY_COMPLEX,
                'color': (0, 0, 255),
                'scale': 1,
                'thickness': 2,
                'mapping': {
                    'text': {
                        'mask': 'MASK ON',
                        'no mask': 'NO MASK'
                    },
                    'color': {
                        'mask': (0, 255, 0),
                        'no mask': (0, 0, 255)
                    }
                }
            }

            if mode == 1:
                detect_all_objects(img, detections[0,0,:,:], scoreThreshold, netModel['classNames'], textLabelSettings)
            elif mode == 2:
                detect_object(img, detections[0,0,:,:], scoreThreshold, netModel['classNames'], currentClassDetecting, detectAllInstances)
            elif mode == 3:
                track_object(img, detections[0,0,:,:], scoreThreshold, netModel['classNames'], currentClassDetecting, trackingThreshold)

        # addObjectCountText(img, currentClassDetecting + ' count: ' + str(totalObjects))
        cv.imshow('Real-Time Object Detection', img)

        if doWriteVideo:
            if videoWriter == None:
                fourcc = cv.VideoWriter_fourcc(*"MJPG")
                videoWriter = cv.VideoWriter(args.output, fourcc, 30, (img.shape[1], img.shape[0]), True)
            videoWriter.write(img)
        
        totalFrames += 1

    print('exiting run_video_detection...')
    if videoWriter != None:
        videoWriter.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net_model", type=int, help="The network model id: \
        0 - MobileNet SSD V1 COCO \
        1 - MobileNet SSD V1 BALLS \
        2 - MobileNet SSD V1 BOXING \
        3 - Inception SSD V2 COCO \
        4 - Face Detector \
        5 - MobileNet SSD V1 MASK/NO MASK" )
    parser.add_argument("detect_mode", type=int, help="The detection mode: \
        1 - detect all objects \
        2 - detect a specific object \
        3 - track a specific object")
    parser.add_argument("-i", "--input", type=str, default=None, help="The path to the optional input video file")
    parser.add_argument("-o", "--output", type=str, default=None, help="The path to the optional output video file")
    parser.add_argument("-f", "--skip_frames", type=int, default=1, help="The number of frames to skip object detection")
    parser.add_argument("-c", "--detect_class", help="The class to detect. Required when mode > 1")
    parser.add_argument("-v", "--voice_cmd", help="Enable voice commands", action="store_true")
    parser.add_argument("-s", "--score_threshold", help="Only show detections with a probability of correctness above the specified threshold", type=float, default=0.3)
    parser.add_argument("-t", "--tracking_threshold", help="Tolerance (delta) between the object being detected and the position it is supposed to be in", type=float, default=50)
    parser.add_argument("-a", "--detect_all_instances", help="When in mode 2, whether or not to detect all instances. If false then detect the highest scored instance", action="store_true")
    parser.add_argument("-l", "--show_labels", action="store_false")
    args = parser.parse_args()

    if args.detect_mode > 1:
        if args.detect_class is None:
            print("Error: You must specify a class to detect if detection mode > 1")
            sys.exit(0)
        else:
            currentClassDetecting = args.detect_class

    doWriteVideo = (args.output != None)
    
    showVideoStream = True
    videoStreamThread = threading.Thread(target=run_video_detection, args=[args.input, args.detect_mode, netModels[args.net_model], args.score_threshold, args.tracking_threshold, args.skip_frames, args.detect_all_instances])
    videoStreamThread.start()

    if args.voice_cmd:
        voiceCommandThread = threading.Thread(target=run_voice_command, args=[netModels[args.net_model]['classNames']])
        voiceCommandThread.start()
