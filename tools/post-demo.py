# -*- coding: utf-8 -*-
"""
@author: lywen
后台通过接口调用服务，获取OCR识别结果
"""
import base64
import requests
import json
def read_img_base64(p):
    with open(p,'rb') as f:
        imgString = base64.b64encode(f.read())
    imgString=b'data:image/jpeg;base64,'+imgString
    return imgString.decode()

def post(p,billModel='generalOCR'):
    URL='http://0.0.0.0:8080/ocr'##url地址
    imgString = read_img_base64(p)
    headers = {}
    param      = {"sessionID": "20190412104001000001",
                "timeStamp": "20190412104001",
                "businessID": "1001",
                "commandID": "100001",
                "imei": "358805090956741",
                "picUrl": "/ocr-photo/mark/201904/17/",
                "picName": "2019041710531449412.jpg",
                  }
    param = json.dumps(param)
    if 1:
            req=requests.post(URL,data= param,headers=None,timeout=20)
            data=req.content
            # data=json.loads(data)
    else:
            data =[]
    print(data)

    
if __name__=='__main__':
    p = '1.jpg'
    post(p,'invoice')