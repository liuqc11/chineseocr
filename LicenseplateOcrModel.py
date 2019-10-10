#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019-10-10 11:19
@Author  : liuqingchen
@Email   : liuqingchen@chinamobile.com
@File    : LicenseplateOcrModel.py
"""
import numpy as np
import cv2
## 用于车牌识别
import darknet.darknet as dn
from darknet.darknet import detect_image
from licenseplate.drawing_utils import draw_label, draw_losangle, write2img
from licenseplate.keras_ocr_utils import LPR
from licenseplate.keras_utils import load_model, detect_lp
from licenseplate.label import Label
from licenseplate.utils import crop_region

YELLOW = (0, 255, 255)
RED = (0, 0, 255)

# vehicle detection model
vehicle_threshold = .5
vehicle_weights = b'darknet/yolov3.weights'
vehicle_netcfg = b'darknet/cfg/yolov3.cfg'
vehicle_dataset = b'darknet/cfg/coco.data'

vehicle_net = dn.load_net_custom(vehicle_netcfg, vehicle_weights, 0, 1)  # batchsize=1
vehicle_meta = dn.load_meta(vehicle_dataset)

# license plate detection model
wpod_net = load_model('models/wpod-net_update1.h5')

# license plate recognition model
ocrmodel = LPR("models/ocr_plate_all_gru.h5")

print("Loaded license plate model!")

def model_lp(img,):

    W, H = img.size
    img = np.asarray(img)
    result_set = set()
    darknet_image = dn.make_image(int(W), int(H), 3)
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dn.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
    # im = nparray_to_image(arr)
    R = detect_image(vehicle_net, vehicle_meta, darknet_image, thresh=vehicle_threshold)
    R = [r for r in R if r[0].decode('utf-8') in ['car', 'bus', 'truck']]
    if len(R):
        WH = np.array(img.shape[1::-1], dtype=float)
        Lcars = []
        for i, r in enumerate(R):

            cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
            tl = np.array([cx - w / 2., cy - h / 2.])
            br = np.array([cx + w / 2., cy + h / 2.])
            label = Label(0, tl, br)
            Lcars.append(label)
            Icar = crop_region(img, label)
            # print('Searching for license plates using WPOD-NET')
            ratio = float(max(Icar.shape[:2])) / min(Icar.shape[:2])
            side = int(ratio * 288.)
            bound_dim = min(side + (side % (2 ** 4)), 608)
            # print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))
            Llp, LlpImgs, _ = detect_lp(wpod_net, Icar / 255, bound_dim, 2 ** 4, (240, 80),
                                        0.5)
            if len(LlpImgs):
                Ilp = LlpImgs[0]
                res, confidence = ocrmodel.recognizeOneframe(Ilp * 255.)

                pts = Llp[0].pts * label.wh().reshape(2, 1) + label.tl().reshape(2, 1)
                ptspx = pts * np.array(img.shape[1::-1], dtype=float).reshape(2, 1)
                draw_losangle(img, ptspx, RED, 3)
                if confidence > 0.5:
                    llp = Label(0, tl=pts.min(1), br=pts.max(1))
                    img = write2img(img, llp, res)
                    result_set.add(res)
        for i, lcar in enumerate(Lcars):
            draw_label(img, lcar, color=YELLOW, thickness=3)
    return img, result_set