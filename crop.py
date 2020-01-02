#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:47:47 2019

@author: Moad Hani
"""

import vision.utils.box_utils_numpy as box_utils

import cv2
from caffe2.python import workspace
import numpy as np


def load_model(init_net_path, predict_net_path):
    with open(init_net_path, "rb") as f:
        init_net = f.read()
    with open(predict_net_path, "rb") as f:
        predict_net = f.read()
    p = workspace.Predictor(init_net, predict_net)
    return p


def predict(width, height, confidences, box, prob_threshold, iou_threshold=0.5, top_k=-1):
    box = box[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_box = box[mask, :]
        box_probs = np.concatenate([subset_box, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                  iou_threshold=iou_threshold,
                                  top_k=top_k,
                                  )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    w_center , h_center = (picked_box_probs[:, 0] +picked_box_probs[:, 2]) / 2. * width,(picked_box_probs[:, 1] +picked_box_probs[:, 3]) / 2. * height
   
#    global w
    h = h_center[0]
    w = w_center[0]
    
    boxw1,boxw2=200, 200
    boxh1,boxh2=250, 150
    img_w,img_h= width,height
    if w> img_w-boxw2:
        w = img_w-boxw2
    if w < boxw1:
        w = boxw1
    
    if h > img_h-boxh2:
        h = img_h-boxh2
    if h< boxh1:
        h =boxh1

    h_center = h
    w_center = w
    picked_box_probs[:, 0] = w_center -boxw1
    picked_box_probs[:, 1] = h_center -boxh1
    picked_box_probs[:, 2] = w_center + boxw2
    picked_box_probs[:, 3] = h_center + boxh2
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


init_net_path = "./models/mobilenet-v1-ssd_init_net.pb"#sys.argv[1]
predict_net_path = "./models/mobilenet-v1-ssd_predict_net.pb"#sys.zrgv[2]
label_path = "./models/voc-model-labels.txt"#sys.argv[3]
class_names = [name.strip() for name in open(label_path).readlines()]
predictor = load_model(init_net_path, predict_net_path)

bs = np.array([[520,280,920,680]])

def crop_dims(image):
    original_image = image
    global bs
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300))
    image = image.astype(np.float32)
    image = (image - 127) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)

    confidences, box = predictor.run({'0': image})
    box, labels, probs = predict(original_image.shape[1], original_image.shape[0], confidences, box, 0.55)

    if box.shape[0] == 0:
        box = bs
    else:
        bs=box
    return box
        

def cropping(image,box):
    
    croped_img = image[ box[0][1]:box[0][3],box[0][0]:box[0][2]]
    return croped_img