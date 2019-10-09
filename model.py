# -*- coding: utf-8 -*-
from config import opencvFlag,GPU,IMGSIZE,ocrFlag
if not GPU:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=''##不启用GPU
    
if ocrFlag=='torch':
    from crnn.crnn_torch import crnnOcr as crnnOcr ##torch版本ocr,目前在使用
elif ocrFlag=='keras':
     from crnn.network_keras import CRNN as crnnOcr ##keras版本OCR
    
import time
import cv2
import numpy as np
from PIL import Image
from glob import glob

from text.detector.detectors import TextDetector
from apphelper.image import get_boxes,letterbox_image

from text.opencv_dnn_detect import angle_detect ##文字方向检测,支持dnn/tensorflow，目前使用keras版本的
from apphelper.image import estimate_skew_angle ,rotate_cut_img,xy_rotate_box,sort_box,box_rotate,solve

## 不同版本的文字检测
if opencvFlag=='opencv':
    from text import opencv_dnn_detect as detect ##opencv dnn model for darknet
elif opencvFlag=='darknet':
    from text import darknet_detect as detect ## darknet版本
else:
    from text import keras_detect as detect ## keras版本，目前在使用

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


print("Text detect engine:{}".format(opencvFlag))

def text_detect(img,
                MAX_HORIZONTAL_GAP=30,
                MIN_V_OVERLAPS=0.6,
                MIN_SIZE_SIM=0.6,
                TEXT_PROPOSALS_MIN_SCORE=0.7,
                TEXT_PROPOSALS_NMS_THRESH=0.3,
                TEXT_LINE_NMS_THRESH = 0.3,
                ):
    boxes, scores = detect.text_detect(np.array(img))
    boxes = np.array(boxes,dtype=np.float32)
    scores = np.array(scores,dtype=np.float32)
    textdetector  = TextDetector(MAX_HORIZONTAL_GAP,MIN_V_OVERLAPS,MIN_SIZE_SIM)
    shape = img.shape[:2]
    boxes = textdetector.detect(boxes,
                                scores[:, np.newaxis],
                                shape,
                                TEXT_PROPOSALS_MIN_SCORE,
                                TEXT_PROPOSALS_NMS_THRESH,
                                TEXT_LINE_NMS_THRESH,
                                )
    
    text_recs = get_boxes(boxes)
    newBox = []
    rx = 1
    ry = 1
    for box in text_recs:
           x1,y1 = (box[0],box[1])
           x2,y2 = (box[2],box[3])
           x3,y3 = (box[6],box[7])
           x4,y4 = (box[4],box[5])
           newBox.append([x1*rx,y1*ry,x2*rx,y2*ry,x3*rx,y3*ry,x4*rx,y4*ry])
    return newBox 



def crnnRec(im,boxes,leftAdjust=False,rightAdjust=False,alph=0.2,f=1.0):
   """
   crnn模型，ocr识别
   leftAdjust,rightAdjust 是否左右调整box 边界误差，解决文字漏检
   """
   results = []
   im = Image.fromarray(im) 
   for index,box in enumerate(boxes):
       degree,w,h,cx,cy = solve(box)
       partImg,newW,newH = rotate_cut_img(im,degree,box,w,h,leftAdjust,rightAdjust,alph)
       text = crnnOcr(partImg.convert('L'))
       if text.strip()!=u'':
            results.append({'cx':cx*f,'cy':cy*f,'text':text,'w':newW*f,'h':newH*f,'degree':degree*180.0/np.pi})
 
   return results




def eval_angle(im,detectAngle=False):
    """
    估计图片偏移角度
    @@param:im
    @@param:detectAngle 是否检测文字朝向
    """
    angle = 0
    img = np.array(im)
    if detectAngle:
        angle = angle_detect(img=np.copy(img))##文字朝向检测
        if angle==90:
            im = Image.fromarray(img).transpose(Image.ROTATE_90)
        elif angle==180:
            im = Image.fromarray(img).transpose(Image.ROTATE_180)
        elif angle==270:
            im = Image.fromarray(img).transpose(Image.ROTATE_270)
        img = np.array(im)
        
    return angle,img


def model(img,detectAngle=False,config={},leftAdjust=False,rightAdjust=False,alph=0.2):
    """
    @@param:img,
    @@param:ifadjustDegree 调整文字识别倾斜角度
    @@param:detectAngle,是否检测文字朝向
    """
    angle,img = eval_angle(img,detectAngle=detectAngle)##文字方向检测
    if opencvFlag!='keras':
       img,f =letterbox_image(Image.fromarray(img), IMGSIZE)## pad
       img = np.array(img)
    else:
        f=1.0##解决box在原图坐标不一致问题
    
    config['img'] = img
    text_recs = text_detect(**config)##文字检测
    newBox = sort_box(text_recs)##行文本识别
    result = crnnRec(np.array(img),newBox,leftAdjust,rightAdjust,alph,1.0/f)
    return img,result,angle


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





