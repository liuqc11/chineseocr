# -*- coding: utf-8 -*-
"""
普通增值税发票
"""
from apphelper.image import union_rbox
import re


class invoice:
    """
    普通增值税发票结构化识别
    """

    def __init__(self, result):
        self.result = result
        # self.result = union_rbox(result, 0.25)
        self.N = len(self.result)
        self.res = {'invoiceCode': '', 'invoiceNo': '', 'invoiceDate': '', 'invoiceAmount': '', 'buyerName': '', 'buyerTaxNo': '',
                    'sellerName': '', 'sellerTaxNo': '', 'picUrl': '', 'picName': ''}
        self.InvoiceCode()
        self.InvoiceNo()
        self.InvoiceDate()
        self.InvoiceAmount()
        self.BuyerName()
        self.BuyerTaxNo()
        self.SellerName()
        self.SellerTaxNo()

    def InvoiceCode(self):
        """
        提取发票代码信息
        :return:self.res.invoicecode
        """
        invoicecode = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ## 发票代码是10位或12位数字（2018年之后）
            res = re.findall('发票代码:(\d{10}|\d{12})', txt)
            if len(res) > 0:
                invoicecode['invoiceCode'] = res[0].replace('发票代码:', '')
                self.res.update(invoicecode)
                break
            res = re.fullmatch('(\d{10}|\d{12})', txt)
            if res:
                invoicecode['invoiceCode'] = txt
                self.res.update(invoicecode)
                break

    def InvoiceNo(self):
        """
        提取发票号码信息
        :return:self.res.invoiceno
        """
        invoiceno = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ## 发票号码一般8位数字
            res = re.findall('发票号码:\d{8}', txt)
            if len(res) > 0:
                invoiceno['invoiceNo'] = res[0].replace('发票号码:', '')
                self.res.update(invoiceno)
                break
            res = re.fullmatch('N0\d{8}', txt)
            if res:
                invoiceno['invoiceNo'] =txt.replace('N0', '')
                self.res.update(invoiceno)
                break
            res = re.fullmatch('\d{8}', txt)
            if res:
                invoiceno['invoiceNo'] = txt
                self.res.update(invoiceno)
                break

    def InvoiceDate(self):
        """
        提取开票日期信息
        :return:self.res.invoicedate
        """
        invoicedate = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ## 开票日期:*年*月*日
            res = re.findall('开票日期:\d{4}年\d{2}月\d{2}日', txt)
            if len(res) > 0:
                invoicedate['invoiceDate'] = res[0].replace('开票日期:', '')
                self.res.update(invoicedate)
                break
            res = re.findall('\d{4}年\d{2}月\d{2}日', txt)
            if len(res) > 0:
                invoicedate['invoiceDate'] = res[0].replace('开票日期:', '')
                self.res.update(invoicedate)
                break

    def InvoiceAmount(self):
        """
        提取发票金额
        :return: self.res.invoiceamount
        """
        invoiceamount = {}
        for i in range(self.N-1, -1, -1):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ## (小写)￥
            res = re.findall('[￥¥]\d*\.\d{2}', txt)
            if len(res) > 0:
                invoiceamount['invoiceAmount'] = res[0]
                self.res.update(invoiceamount)
                break
            res = re.findall('[零壹贰叁肆伍陆柒捌玖拾佰仟万亿]*圆[整零壹贰叁肆伍陆柒捌玖拾角分]*', txt)
            if len(res) > 0:
                invoiceamount['invoiceAmount'] = res[0].replace('(小写)¥', '')
                self.res.update(invoiceamount)
                break
            # print(resAlpha)

    def BuyerName(self):
        """
        受票方名称
        :return:self.res.buyername
        """
        buyername = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ## 称:
            ## 《中华人民共和国发票管理办法实施细则》第三十六条的规定：开具发票应当使用中文。
            ## 民族自治地方可以同时使用当地通用的一种民族文字。
            ## 外商投资企业和外国企业可以同时使用一种外国文字。
            res = re.findall('称:[\u4e00-\u9fa5]*', txt)
            if len(res) > 0:
                buyername['buyerName'] = res[0].replace('称:', '')
                self.res.update(buyername)
                break

    def BuyerTaxNo(self):
        """
        受票方税号
        :return:self.res.buyertaxno
        """
        buyertaxno = {}
        for i in range(self.N):
            txt = self.result[i]['text']
            # txt = txt.replace(' ', '')
            ## 纳税人识别号:
            res = re.findall('纳税人识别号:[a-zA-Z0-9]{18}', txt)
            if len(res) > 0:
                buyertaxno['buyerTaxNo'] = res[0].replace('纳税人识别号:', '')
                self.res.update(buyertaxno)
                break
            # res = re.findall('(?!N0)[a-zA-Z0-9]{18}', txt)
            # if len(res) > 0:
            #     buyertaxno['BuyerTaxNo'] = res[0].replace('纳税人识别号:', '')
            #     self.res.update(buyertaxno)
            #     break

    def SellerName(self):
        """
        销售方名称
        :return:self.res.sellername
        """
        sellername = {}
        for i in range(self.N - 1, -1, -1):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ## 称:
            ## 《中华人民共和国发票管理办法实施细则》第三十六条的规定：开具发票应当使用中文。
            ## 民族自治地方可以同时使用当地通用的一种民族文字。
            ## 外商投资企业和外国企业可以同时使用一种外国文字。
            res = re.findall('称:[\(\)\u4e00-\u9fa5]*', txt)
            if len(res) > 0:
                sellername['sellerName'] = res[0].replace('称:', '')
                self.res.update(sellername)
                break

    def SellerTaxNo(self):
        """
        销售方税号
        :return:self.res.sellertaxno
        """
        sellertaxno = {}
        for i in range(self.N - 1, -1, -1):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ## 纳税人识别号:
            res = re.findall('纳税人识别号:[a-zA-Z0-9]{18}', txt)
            if len(res) > 0:
                sellertaxno['sellerTaxNo'] = res[0].replace('纳税人识别号:', '')
                self.res.update(sellertaxno)
                break
            res = re.fullmatch('[a-zA-Z0-9]{18}', txt)
            if res:
                sellertaxno['sellerTaxNo'] = txt
                self.res.update(sellertaxno)
                break
