#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019-10-10 11:19
@Author  : liuqingchen
@Email   : liuqingchen@chinamobile.com
@File    : LicenseplateOcrModel.py
"""
import numpy as np
from  PIL import Image
import cv2
## 用于车牌识别
import darknet.darknet as dn
from darknet.darknet import detect_image
from licenseplate.drawing_utils import draw_label, draw_losangle, write2img
from licenseplate.keras_ocr_utils import LPR
from licenseplate.keras_utils import load_model, detect_lp
from licenseplate.label import Label
from licenseplate.utils import crop_region

## MTCNN的车牌模型
from config import GPU
import torch
from licenseplate.MTCNN.MTCNN import create_mtcnn_net
from licenseplate.LPRNet.model.STN import STNet
from licenseplate.LPRNet.model.LPRNET import LPRNet, CHARS
from licenseplate.LPRNet.LPRNet_Test import cv2ImgAddText,decode

if torch.cuda.is_available() and GPU:
    device = "cuda:0"
else:
    device = "cpu"

YELLOW = (0, 255, 255)
RED = (0, 0, 255)


class LicenseplateOcrModel(object):
    # def __init__(self, vehicle_weights, vehicle_netcfg, vehicle_dataset, licensemodel, ocrmodel)
    def __init__(self):
        # self.vehicle_net = dn.load_net_custom(vehicle_netcfg, vehicle_weights, 0, 1)  # batchsize=1
        # self.vehicle_threshold = .5
        # self.vehicle_meta = dn.load_meta(vehicle_dataset)
        # self.license_model = load_model(licensemodel)
        # self.ocr_model = LPR(ocrmodel)

        ## MTCNN

        # STN and LPRNet
        self.STN = STNet()
        self.STN.to(device)
        self.STN.load_state_dict(torch.load('./licenseplate/LPRNet/weights/Final_STN_model.pth',
                                       map_location=lambda storage, loc: storage))
        self.STN.eval()

        self.lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
        self.lprnet.to(device)
        self.lprnet.load_state_dict(
            torch.load('./licenseplate/LPRNet/weights/Final_LPRNet_model.pth', map_location=lambda storage, loc: storage))
        self.lprnet.eval()


    def model(self, img: Image) -> (np.array, set):
        W, H = img.size
        img = np.asarray(img)
        result_set = set()
        darknet_image = dn.make_image(int(W), int(H), 3)
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dn.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
        # im = nparray_to_image(arr)
        R = detect_image(self.vehicle_net, self.vehicle_meta, darknet_image, thresh=self.vehicle_threshold)
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
                Llp, LlpImgs, _ = detect_lp(self.license_model, Icar / 255, bound_dim, 2 ** 4, (240, 80),
                                            0.5)
                if len(LlpImgs):
                    Ilp = LlpImgs[0]
                    res, confidence = self.ocr_model.recognizeOneframe(Ilp * 255.)

                    pts = Llp[0].pts * label.wh().reshape(2, 1) + label.tl().reshape(2, 1)
                    ptspx = pts * np.array(img.shape[1::-1], dtype=float).reshape(2, 1)
                    draw_losangle(img, ptspx, RED, 3)
                    if confidence > 0.5:
                        llp = Label(0, tl=pts.min(1), br=pts.max(1))
                        img = write2img(img, llp, res)
                        result_set.add(res)
            for i, lcar in enumerate(Lcars):
                draw_label(img, lcar, color=YELLOW, thickness=3)
        # 如果没检测到车，就直接检测-识别车牌(需要改进)
        # ToDO
        else:
            # print('Searching for license plates using WPOD-NET')
            ratio = float(max(W,H)) / min(W,H)
            side = int(ratio * 288.)
            bound_dim = min(side + (side % (2 ** 4)), 608)
            # print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))
            Llp, LlpImgs, _ = detect_lp(self.license_model, img / 255.0, bound_dim, 2 ** 4, (240, 80),
                                        0.5)
            if len(LlpImgs):
                Ilp = LlpImgs[0]
                res, confidence = self.ocr_model.recognizeOneframe(Ilp * 255.)
                ptspx = Llp[0].pts * np.array(img.shape[1::-1], dtype=float).reshape(2, 1)
                draw_losangle(img, ptspx, RED, 3)
                if confidence > 0.5:
                    llp = Label(0, tl=Llp[0].pts.min(1), br=Llp[0].pts.max(1))
                    img = write2img(img, llp, res)
                    result_set.add(res)
            # 没检测到车牌，就直接尝试识别车牌
            else:
                res, confidence = self.ocr_model.recognizeOneframe(img)
                if confidence > 0.5:
                    result_set.add(res)
        return img, result_set

    def model_video(self, video_path, output_path):
        vid = cv2.VideoCapture(video_path)
        video_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWriter = cv2.VideoWriter(output_path, fourcc, int(video_fps//5),
                                      (int(video_width), int(video_height)))
        result = set()
        frame_count = 0
        while True:
            return_value, arr = vid.read()
            if not return_value:
                break
            if frame_count % 5 == 0:
                tmpimage = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                tmpimage, result_set = self.model_MTCNN(tmpimage)
                result.update(result_set)
                newimage = cv2.cvtColor(np.asarray(tmpimage),cv2.COLOR_RGB2BGR)
                videoWriter.write(newimage)
            else:
                pass
            frame_count = frame_count+1
        videoWriter.release()
        return result

    def model_MTCNN(self, img) -> (np.array,set):
        """
        使用MTCNN-LPRNet进行车牌识别
        :param img: PIL读取的图像格式
        :return: img是np.array格式或者OpenCV支持读取的图像，识别的车牌文字集合result_set
        """
        # W, H = img.size
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        result_set = set()
        mini_lp = (50,15) ## Minimum lp to be detected. derease to increase accuracy. Increase to increase speed
        ## MTCNN
        bboxes = create_mtcnn_net(img,mini_lp,device,p_model_path="./licenseplate/MTCNN/weights/pnet_Weights",
                                  o_model_path="./licenseplate/MTCNN/weights/onet_Weights")
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :4]
            x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]
            w = int(x2 - x1 + 1.0)
            h = int(y2 - y1 + 1.0)
            img_box = np.zeros((h, w, 3))
            img_box = img[y1:y2 + 1, x1:x2 + 1, :]
            im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
            im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
            data = torch.from_numpy(im).float().unsqueeze(0).to(device)
            transfer = self.STN(data)
            preds = self.lprnet(transfer)
            preds = preds.cpu().detach().numpy()
            labels, pred_labels = decode(preds, CHARS)
            # print(labels[0])
            result_set.add(labels[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            img = cv2ImgAddText(img, labels[0], (x1, y1 - 12), textColor=(255, 255, 0), textSize=15)


        return img, result_set



