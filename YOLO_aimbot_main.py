import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import subprocess
import time
from datetime import datetime
import win32gui
import win32process
import cv2
import mss
import numpy as np
import tensorflow as tf
from yolov3.utils import *
from yolov3.configs import *
from yolov3.yolov4 import read_class_names
from tools.Detection_to_XML import CreateXMLfile
import random

def find_enemy(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):   
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    detection_list = []

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        x, y = int(x1+(x2-x1)/2), int(y1+(y2-y1)/2)
        detection_list.append([NUM_CLASS[class_ind], x, y])

        
    return image, detection_list

def detect_enemy(Yolo, original_image, input_size=416, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    image_data = image_preprocess(original_image, [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)

    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)
        
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    image, detection_list = find_enemy(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        
    return image, detection_list, bboxes

def get_hwnds_forpid(pid):
    def callback(hwnd, hwnds):
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
        if found_pid == pid:
            hwnds.append(hwnd)
        return True
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    return hwnds

def getwindowgeometry(pid):
    while True:
        handles = get_hwnds_forpid(pid)
        for i in handles:
            inFocus = win32gui.GetActiveWindow()
            if i != inFocus:
                win32gui.SetForegroundWindow(i)
                time.sleep(2)
            left, top, right, bottom = win32gui.GetWindowRect(i)
            if (left, top, right, bottom) != (0, 0, 0, 0):
                return left, top, right-left, bottom-top

def print_data(bboxes, detection_list, log_file):
    for i in range(min(len(bboxes), len(detection_list))):
        print(datetime.now(), 'accuracy = %.2f%%' % (bboxes[i][4] * 100), detection_list[i])
        print(datetime.now(), 'accuracy = %.2f%%' % (bboxes[i][4] * 100), detection_list[i], file=log_file)

def main():
    offset = 30
    times = []
    sct = mss.mss()
    yolo = Load_Yolo_model()
    x, y, w, h = getwindowgeometry(int(sys.argv[1]))

    with open(sys.argv[2], 'w') as log_file:
        while True:
            t1 = time.time()
            img = np.array(sct.grab({"top": y, "left": x, "width": w, "height": h, "mon": -1}))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            image, detection_list, bboxes = detect_enemy(yolo, np.copy(img), input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
            if detection_list != []:
                print_data(bboxes, detection_list, log_file)

            t2 = time.time()
            times.append(t2-t1)
            times = times[-50:]
            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            print("FPS", fps)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Wrong usage, please enter in format 'python YOLO_aimbot_main.py <PID> <log file path>'")
        exit(1)
    main()