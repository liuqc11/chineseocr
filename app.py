# -*- coding: utf-8 -*-
"""
@author: liuqingchen
"""
import os
import cv2
import json
import time
import uuid
import web
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from apphelper.MyLog import Log
web.config.debug= False

filelock='file.lock'
if os.path.exists(filelock):
   os.remove(filelock)

# import model
render = web.template.render('templates', base='base')
from config import *
from apphelper.image import union_rbox,adjust_box_to_origin,xy_rotate_box, box_rotate,base64_to_PIL
from application import trainTicket,idcard,invoice,bankcard

if yoloTextFlag == 'keras' or AngleModelFlag == 'tf' or ocrFlag == 'keras':
    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf
        from keras import backend as K

        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.15  ## GPU最大占用量，测试预留8G显存较好
        config.gpu_options.allow_growth = False  ##GPU是否可动态增加
        K.set_session(tf.Session(config=config))
        K.get_session().run(tf.global_variables_initializer())

    else:
        ##CPU启动
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

if yoloTextFlag == 'opencv':
    scale, maxScale = IMGSIZE
    from text.opencv_dnn_detect import text_detect
elif yoloTextFlag == 'darknet':
    scale, maxScale = IMGSIZE
    from text.darknet_detect import text_detect
elif yoloTextFlag == 'keras':
    scale, maxScale = IMGSIZE[0], 2048
    from text.keras_detect import text_detect
else:
    print("err,text engine in keras\opencv\darknet")

from text.opencv_dnn_detect import angle_detect

if ocr_redis:
    ##多任务并发识别
    from apphelper.redisbase import redisDataBase

    ocr = redisDataBase().put_values
else:
    from crnn.keys import alphabetChinese, alphabetEnglish

    if ocrFlag == 'keras':
        from crnn.network_keras import CRNN

        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelKerasLstm
            else:
                ocrModel = ocrModelKerasDense
        else:
            ocrModel = ocrModelKerasEng
            alphabet = alphabetEnglish
            LSTMFLAG = True

    elif ocrFlag == 'torch':
        from crnn.network_torch import CRNN

        if chineseModel:
            alphabet = alphabetChinese
            if LSTMFLAG:
                ocrModel = ocrModelTorchLstm
            else:
                ocrModel = ocrModelTorchDense

        else:
            ocrModel = ocrModelTorchEng
            alphabet = alphabetEnglish
            LSTMFLAG = True
    elif ocrFlag == 'opencv':
        from crnn.network_dnn import CRNN

        ocrModel = ocrModelOpencv
        alphabet = alphabetChinese
    else:
        print("err,ocr engine in keras\opencv\darknet")

    nclass = len(alphabet) + 1
    if ocrFlag == 'opencv':
        crnn = CRNN(alphabet=alphabet)
    else:
        crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet)
    if os.path.exists(ocrModel):
        crnn.load_weights(ocrModel)
    else:
        print("download model or tranform model with tools!")

    ocr = crnn.predict_job

from TextOcrModel import TextOcrModel
model = TextOcrModel(ocr,text_detect,angle_detect)

from LicenseplateOcrModel import LicenseplateOcrModel
vehicle_weights = b'darknet/yolov3.weights'
vehicle_netcfg = b'darknet/cfg/yolov3.cfg'
vehicle_dataset = b'darknet/cfg/coco.data'
licensemodel = 'models/wpod-net_update1.h5'
ocrmodel = 'models/ocr_plate_all_gru.h5'
model_lp = LicenseplateOcrModel(vehicle_weights, vehicle_netcfg, vehicle_dataset, licensemodel, ocrmodel)


billList = ['general_OCR', 'trainticket', 'idcard', 'invoice', 'bankcard', 'licenseplate']

class OCR:
    """通用OCR识别"""
    def __init__(self, ):
        self.logger = web.ctx.environ['wsgilog.logger']  # 使用日志 #

    def plot_box(self, img, boxes):
        blue = (0, 0, 0)  # 18
        tmp = np.copy(img)
        for box in boxes:
            cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), blue, 1)  # 19

        return Image.fromarray(tmp)

    def plot_boxes(self, img, angle, result, color=(0, 0, 0)):
        tmp = np.array(img)
        c = color
        h, w = tmp.shape[:2]
        thick = int((h + w) / 300)
        i = 0
        if angle in [90, 270]:
            imgW, imgH = tmp.shape[:2]

        else:
            imgH, imgW = tmp.shape[:2]

        for line in result:
            cx = line['cx']
            cy = line['cy']
            degree = line['degree']
            w = line['w']
            h = line['h']

            x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(cx, cy, w, h, degree / 180 * np.pi)

            x1, y1, x2, y2, x3, y3, x4, y4 = box_rotate([x1, y1, x2, y2, x3, y3, x4, y4], angle=(360 - angle) % 360,
                                                        imgH=imgH, imgW=imgW)
            cx = np.mean([x1, x2, x3, x4])
            cy = np.mean([y1, y2, y3, y4])
            cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)
            cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), c, 1)
            cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), c, 1)
            cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), c, 1)
            mess = str(i)
            cv2.putText(tmp, mess, (int(cx), int(cy)), 0, 1e-3 * h, c, thick // 2)
            i += 1

        return Image.fromarray(tmp)

    def format_text(self, textbox, img, angle, billModel='general_OCR', CommandID= ''):
        """
        格式化各种图片提取的文本
        :param textbox: 提取的文本框（包括坐标和文本内容）
        :param img: 原图
        :param angle: 原图需要旋转的角度
        :param billModel: 图片类型，方便格式化
        :param CommandID: 判断来自网页的请求（文本展示），还是返回给服务器的请求
        :return: res: json格式的格式化结果
        """
        if billModel == '' or billModel == 'general_OCR':
            result = union_rbox(textbox, 0.2)
            res = [{'text': x['text'],
                    'name': str(i),
                    'box': {'cx': x['cx'],
                            'cy': x['cy'],
                            'w': x['w'],
                            'h': x['h'],
                            'angle': x['degree']

                            }
                    } for i, x in enumerate(result)]
            res = adjust_box_to_origin(img, angle, res)  ##修正box
        elif billModel == 'trainticket':
            res = trainTicket.trainTicket(textbox)
            res = res.res
            if CommandID != '':
                res = {key: res[key] for key in res}
            else:
                res = [{'text': res[key], 'name': key, 'box': {}} for key in res]
        elif billModel == 'idcard':
            res = idcard.idcard(textbox)
            res = res.res
            if CommandID != '':
                res = {key: res[key] for key in res}
            else:
                res = [{'text': res[key], 'name': key, 'box': {}} for key in res]
        elif billModel == 'invoice':
            res = invoice.invoice(textbox)
            res = res.res
            if CommandID != '':
                res = {key: res[key] for key in res}
            else:
                res = [{'text': res[key], 'name': key, 'box': {}} for key in res]
        elif billModel == 'bankcard':
            res = bankcard.bankcard(textbox)
            res = res.res
            if CommandID != '':
                res = {key: res[key] for key in res}
            else:
                res = [{'text': res[key], 'name': key, 'box': {}} for key in res]
        elif billModel == 'licenseplate':
            if CommandID != '':
                res = {'carNo': list(textbox), 'picUrl': '', 'picName': ''}
            else:
                res = [{'text': text, 'name': 'carNo', 'box': {}} for text in list(textbox)]

        return res

    def GET(self):
        self.logger.info('request to open the ocr page.')
        post = {}
        post['postName'] = 'ocr'
        post['height'] = 1920
        post['H'] = 1920
        post['width'] = 1080
        post['W'] = 1080
        post['uuid'] = uuid.uuid1().__str__()
        post['billList'] = billList
        return render.ocr(post)

    def POST(self):
        data = web.data()
        data = json.loads(data)
        CommandID = data.get('commandID', '')
        BusinessID = data.get('businessID','')
        SessionID = data.get('sessionID','')

        # 下面三行兼容原有的web app demo
        billModel = data.get('billModel','') ## 确定具体使用哪种模式识别
        textAngle = data.get('textAngle', True)  ## 文字方向检测
        textLine = data.get('textLine', False)  ## 只进行单行识别

        # 处理传递参数
        if CommandID != '':
            self.logger.info('post request from JiuTian IP= %s ,CommandID=%s' % (web.ctx.get('ip'), CommandID))
            if CommandID == '100001':
                billModel = 'invoice'
            elif CommandID == '200001':
                billModel = 'idcard'
            elif CommandID == '300001':
                billModel = 'bankcard'
            elif CommandID == '400001':
                billModel = 'licenseplate'
            else:
                ## 返回请求参数错误
                return json.dumps(
                    {'sessionID': SessionID,
                     'commandID': CommandID,
                     'businessID': BusinessID,
                     'timeStamp': time.strftime('%Y%m%d%H%M%S', time.localtime()),
                     'execStatus': {"statusCode": 0x800003, "statusDescription": "请求参数错误"},
                     'resultInfo': {}}, ensure_ascii=False
                )
            picName = data.get('picName', 'new.jpg')
            picpath = 'http://172.31.201.35:18081' + data.get('picUrl', '') + picName
            response = requests.get(picpath, stream=True)
            ## 处理可能出现的视频（只可能出现在‘licenseplate’中）
            if picName.endswith(('.jpg', '.png', '.jpeg', '.JPG','.JPEG','.PNG')):
                img = Image.open(BytesIO(response.content)).convert('RGB')
            elif picName.endswith(('.mp4','.MP4','.avi','.AVI')) and billModel == 'licenseplate':
                with open(picName, 'wb+') as f:
                    f.write(response.content)
                saveName = picName.split('.')[0]+'_new.mp4'
                result = model_lp.model_video(picName,saveName)
                res = {'carNo': list(result), 'picUrl': '', 'picName': ''}
                upload_url = 'http://172.31.201.35:18081' + '/cmcc-ocr-webapi-1.0/service/remoteUploadPic/'
                files = {'image': (saveName, open(saveName, 'rb'), 'image/jpeg', {})}
                reply = requests.post(upload_url, files=files)
                # get the picUrl and picName
                reply = reply.json()
                # print(reply)
                res['picUrl'] = reply['picUrl']
                res['picName'] = reply['picName']
                # delete tmp files
                # os.remove(picName)
                #os.remove(saveName)

                return json.dumps({'sessionID': SessionID,
                                   'commandID': CommandID,
                                   'businessID': BusinessID,
                                   'timeStamp': time.strftime('%Y%m%d%H%M%S', time.localtime()),
                                   'execStatus': {"statusCode": 0x000000, "statusDescription": "成功"},
                                   'resultInfo': res}, ensure_ascii=False)
            else:
                ## 返回请求参数错误
                return json.dumps(
                    {'sessionID': SessionID,
                     'commandID': CommandID,
                     'businessID': BusinessID,
                     'timeStamp': time.strftime('%Y%m%d%H%M%S', time.localtime()),
                     'execStatus': {"statusCode": 0x800004, "statusDescription": "内部数据错误"},
                     'resultInfo': {}}, ensure_ascii=False
                )
        else:
            ## 兼容原有的web app demo
            imgString = data['imgString'].encode().split(b';base64,')[-1]
            img = base64_to_PIL(imgString)

        if img is not None:
            img = np.array(img)

        H, W = img.shape[:2]
        timeTake = time.time()
        if textLine:
            ##单行识别
            partImg = Image.fromarray(img)
            # text = model.crnnOcr(partImg.convert('L'))
            text = crnn.predict(partImg.convert('L'))
            res = [{'text': text, 'name': '0', 'box': [0, 0, W, 0, W, H, 0, H]}]
        else:
            if billModel == 'licenseplate':
                img = Image.fromarray(img)
                img, result = model_lp.model(img)
                res = self.format_text(result, img, 0, billModel, CommandID)
            else:
                detectAngle = textAngle
                result, angle = model.model(img,
                                            scale=scale,
                                            maxScale=maxScale,
                                            detectAngle=detectAngle,  ##是否进行文字方向检测，通过web传参控制
                                            MAX_HORIZONTAL_GAP=100,  ##字符之间的最大间隔，用于文本行的合并
                                            MIN_V_OVERLAPS=0.6,
                                            MIN_SIZE_SIM=0.6,
                                            TEXT_PROPOSALS_MIN_SCORE=0.1,
                                            TEXT_PROPOSALS_NMS_THRESH=0.3,
                                            TEXT_LINE_NMS_THRESH=0.99,  ##文本行之间测iou值
                                            LINE_MIN_SCORE=0.1,
                                            leftAdjustAlph=0.01,  ##对检测的文本行进行向左延伸
                                            rightAdjustAlph=0.01,  ##对检测的文本行进行向右延伸
                                            )
                res = self.format_text(result, img, angle, billModel, CommandID)

        timeTake = time.time() - timeTake

        ## 输出，同样区分是否是原有的web app demo接口
        if CommandID == '':
            # os.remove(path)
            # print(res)
            # outpic = self.plot_boxes(img, angle, result, color=(0, 0, 0))
            # outpic.save('new.jpg')
            return json.dumps({'res': res, 'timeTake': round(timeTake, 4)}, ensure_ascii=False)
        else:
            if timeTake > 15:
                return json.dumps(
                    {'sessionID': SessionID,
                     'commandID': CommandID,
                     'businessID': BusinessID,
                     'timeStamp': time.strftime('%Y%m%d%H%M%S', time.localtime()),
                     'execStatus': {"statusCode": 0x800001, "statusDescription": "响应超时"},
                     'resultInfo': {}}, ensure_ascii=False
                )
            # save and upload the box pic
            if billModel == 'licenseplate':
                outpic = Image.fromarray(img)
            else:
                outpic = self.plot_boxes(img, angle, result, color=(0, 0, 0))
            outpic.save(picName)
            upload_url = 'http://172.31.201.35:18081' + '/cmcc-ocr-webapi-1.0/service/remoteUploadPic/'
            files = {'image': (picName, open(picName, 'rb'), 'image/jpeg', {})}
            reply = requests.post(upload_url, files=files)
            # get the picUrl and picName
            reply = reply.json()
            # print(reply)
            res['picUrl'] = reply['picUrl']
            res['picName'] = reply['picName']
            # delete tmp files
            os.remove(picName)

            return json.dumps({'sessionID': SessionID,
                               'commandID': CommandID,
                               'businessID': BusinessID,
                               'timeStamp': time.strftime('%Y%m%d%H%M%S', time.localtime()),
                               'execStatus': {"statusCode": 0x000000, "statusDescription": "成功"},
                               'resultInfo': res}, ensure_ascii=False)

        

urls = ('/ocr', 'OCR',)

if __name__ == "__main__":
      app = web.application(urls, globals())
      app.run(Log)
