#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020-06-02 17:25
@Author  : liuqingchen
@Email   : liuqingchen@chinamobile.com
@File    : Test_KerasDetectText.py.py
"""
import sys
sys.path.append("..")
from config import *
import cv2
from PIL import Image
import numpy as np
import os

if GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
    import tensorflow as tf
    from keras import backend as K

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.1  ## GPU最大占用量，测试预留8G显存较好
    config.gpu_options.allow_growth = False  ##GPU是否可动态增加
    K.set_session(tf.Session(config=config))
    K.get_session().run(tf.global_variables_initializer())

else:
    ##CPU启动
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

scale, maxScale = IMGSIZE[0], 2048
from text.keras_detect import text_detect


def plot_box(img, boxes):
    blue = (0, 0, 0)  # 18
    tmp = np.copy(img)
    for box in boxes:
        cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), blue, 1)  # 19

    return Image.fromarray(tmp)

def test_keras_detect(img, **args):
    scale = args.get('scale', 608)
    maxScale = args.get('maxScale', 608)
    boxes, scores = text_detect(img, scale, maxScale, prob=0.06)
    tmp = plot_box(img, boxes)
    tmp.save('./test.jpg')


if __name__ == '__main__':
    img = Image.open('./new.jpg')
    img = np.array(img)
    test_keras_detect(img,scale=scale,maxScale=maxScale)
