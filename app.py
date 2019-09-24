# -*- coding: utf-8 -*-
"""
@author: liuqingchen
"""
import os
import cv2
import json
import time
import uuid
import base64
import web
import numpy as np
import requests
from PIL import Image
from io import BytesIO
web.config.debug= False
import model
render = web.template.render('templates', base='base')
# from config import DETECTANGLE
from apphelper.image import union_rbox,adjust_box_to_origin,xy_rotate_box, box_rotate
from application import trainTicket,idcard,invoice,bankcard


billList = ['general_OCR', 'trainticket', 'idcard', 'invoice', 'bankcard', 'licenseplate']

class OCR:
    """通用OCR识别"""

    def plot_box(self, img, boxes):
        blue = (0, 0, 0)  # 18
        tmp = np.copy(img)
        for box in boxes:
            cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), blue, 1)  # 19

        return Image.fromarray(tmp)

    def plot_boxes(self, img, angle, result, color=(0, 0, 0)):
        tmp = np.array(img)
        c = color
        w , h = img.size
        thick = int((h + w) / 300)
        i = 0
        if angle in [90, 270]:
            imgH, imgW = img.size

        else:
            imgW, imgH = img.size

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

        return res

    def GET(self):
        post = {}
        post['postName'] = 'ocr'##请求地址
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
            picpath = 'http://172.29.73.70:8099' + data.get('picUrl', '') + picName
            if picName.endswith(('.jpg','.png','.jpeg','.bmp')):
                response = requests.get(picpath)
                img = Image.open(BytesIO(response.content)).convert('RGB')
            elif picName.endswith(('.mp4','.avi')):
                with requests.get(picpath, stream=True) as r:
                    with open(picName,'ab+') as f:
                        f.write(r.content)
            else:
                ## 返回内部数据错误
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
            imgString = base64.b64decode(imgString)
            jobid = uuid.uuid1().__str__()
            path = 'test/{}.jpg'.format(jobid)
            with open(path,'wb') as f:
                f.write(imgString)
            img = Image.open(path).convert('RGB')##GBR

        if billModel == 'licenseplate':
            pass
        else:
            W, H = img.size
            timeTake = time.time()
            if textLine:
                ##单行识别
                partImg = Image.fromarray(img)
                text = model.crnnOcr(partImg.convert('L'))
                res = [{'text': text, 'name': '0', 'box': [0, 0, W, 0, W, H, 0, H]}]

            else:
                detectAngle = textAngle
                _, result, angle = model.model(img,
                                               detectAngle=detectAngle,  ##是否进行文字方向检测，通过web传参控制
                                               config=dict(MAX_HORIZONTAL_GAP=50,  ##字符之间的最大间隔，用于文本行的合并
                                                           MIN_V_OVERLAPS=0.6,
                                                           MIN_SIZE_SIM=0.6,
                                                           TEXT_PROPOSALS_MIN_SCORE=0.1,
                                                           TEXT_PROPOSALS_NMS_THRESH=0.3,
                                                           TEXT_LINE_NMS_THRESH=0.7,  ##文本行之间测iou值
                                                           ),
                                               leftAdjust=True,  ##对检测的文本行进行向左延伸
                                               rightAdjust=True,  ##对检测的文本行进行向右延伸
                                               alph=0.01,  ##对检测的文本行进行向右、左延伸的倍数
                                               )
                res = self.format_text(result, img, angle, billModel, CommandID)

            timeTake = time.time() - timeTake
            if CommandID == '':
                os.remove(path)
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
                outpic = self.plot_boxes(img, angle, result, color=(0, 0, 0))
                outpic.save(picName)
                upload_url = 'http://172.29.73.70:8099' + '/cmcc-ocr-webapi-1.0/service/remoteUploadPic/'
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
      app.run()
