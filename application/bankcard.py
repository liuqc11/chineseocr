# -*- coding: utf-8 -*-
"""
银行卡
"""
from apphelper.image import union_rbox
import re


class bankcard:
    """
    银行卡结构化识别
    """
    def __init__(self, result):
        # self.result = result
        self.result = union_rbox(result, 0.3)
        self.N = len(self.result)
        self.res = {'bankCardNo': '', 'expiryDate': '', 'picUrl': '', 'picName': ''}
        self.BankCardNo()
        self.ExpiryDate()

    def BankCardNo(self):
        """
        提取银行卡号信息
        :return:self.res.bankcardno
        """
        bankcardno = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ## 银行卡号一般是19位的，像工商银行、农业银行等都是19位的，而招商银行、建设银行是16位，交通银行是17位
            ## 信用卡的位数一般都是16位
            res = re.findall('[bB\.\d]{16,19}', txt)
            if len(res) > 0:
                bankcardno['bankCardNo'] = res[0]
                self.res.update(bankcardno)
                break

    def ExpiryDate(self):
        """
        有效期
        :return:self.res.expirydate
        """
        expirydate = {}
        for i in range(self.N-1,-1,-1):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            res = re.findall('RU\d{2}[17\\\/]\d{2}', txt)
            if len(res) > 0:
                expirydate['expiryDate'] = res[0].replace('RU','')[0:2]+'/'+res[0].replace('RU','')[3:5]
                self.res.update(expirydate)
                break
            res = re.findall('\d{2}[17\\\/]\d{2}', txt)
            if len(res):
                expirydate['expiryDate'] = res[0][0:2]+'/'+res[0][3:5]
                self.res.update(expirydate)
                break