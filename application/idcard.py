# -*- coding: utf-8 -*-
"""
身份证
"""
from apphelper.image import union_rbox
import re


class idcard:
    """
    身份证结构化识别
    """
    def __init__(self,result):
        self.result = union_rbox(result,0.2)
        self.N = len(self.result)
        self.res = {'name': '', 'gender': '', 'ethnicity': '', 'birthday': '', 'idNumber': '', 'address': '',
                    'authority': '', 'effectiveDate': '', 'expiryDate': '', 'picUrl': '', 'picName': ''}
        self.full_name()
        self.birthNo()
        self.gender()
        self.ethnicity()
        self.birthday()
        self.address()
        self.authority()
        self.effecttime()
        
    
    def full_name(self):
        """
        身份证姓名
        """
        name={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            ##匹配身份证姓名
            res = re.findall("姓名[\u4e00-\u9fa5]{1,7}",txt)
            if len(res)>0:
                name['name']=res[0].replace('姓名','')
                self.res.update(name) 
                break
        if self.res['name'] == '':
            self.res['name'] = self.result[0]['text'].replace(' ','')

    def gender(self):
        """
        性别
        :return: self.res.update(gender)
        """
        gender={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ##性别女
            res = re.findall(".*性别[男女]+", txt)
            if len(res) > 0:
                gender["gender"] = res[0].split('性别')[-1]
                self.res.update(gender)
                break
        if '男' not in self.res['gender'] and '女' not in self.res['gender']:
            if self.res['idNumber'][-2] in {'1','3','5','7','9'}:
                self.res['gender'] = '男'
            else:
                self.res['gender'] = '女'

    def ethnicity(self):
        """
        民族汉
        """
        ethnicity={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            ##民族汉
            res = re.findall(".*民族[\u4e00-\u9fa5]+",txt)
            if len(res)>0:
                ethnicity["ethnicity"] = res[0].split('民族')[-1]
                self.res.update(ethnicity)
                break

    def birthday(self):
        """
        出生年月
        """
        birth={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            ##出生年月
            res = re.findall('出生\d+年\d+月\d+日',txt)
            res = re.findall('\d+年\d+月\d+日',txt)
            
            if len(res)>0:
                birth['birthday']  =res[0].replace('出生','').replace('年','-').replace('月','-').replace('日','')
                self.res.update(birth) 
                break
        if self.res['birthday'] == '':
            self.res['birthday'] = self.res['idNumber'][6:10]+'-'+ \
                                   self.res['idNumber'][10:12]+'-'+self.res['idNumber'][12:14]
                
    def birthNo(self):
        """
        身份证号码
        """
        No={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            ##身份证号码
            res = re.findall('号码\d{17}[X|x]',txt)
            res = re.findall('号码\d{18}',txt)
            res = re.findall('\d{17}[X|x]',txt)
            res = re.findall('\d{18}',txt)
            
            if len(res)>0:
                No['idNumber']=res[0].replace('号码','')
                self.res.update(No) 
                break    
                
    def address(self):
        """
        身份证地址
        ##此处地址匹配还需完善
        """
        add={}
        addString=[]
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            
            ##身份证地址
            if '住址' in txt or '省' in txt or '市' in txt or '自治区' in txt or '街' in txt or '村' in txt or "镇" in txt or "区" in txt or "城" in txt:
                addString.append(txt.replace('住址',''))
                txt_cy = self.result[i]['cy']
                txt_h = self.result[i]['h']
                for j in range(i+1,self.N):
                    next_pos_cy = self.result[j]['cy']
                    next_txt = self.result[j]['text'].replace(' ','')
                    if next_pos_cy <= (txt_cy + 1.4*(txt_h)):
                        addString.append(next_txt.replace('住址',''))
                    else:
                        break
                    txt_h = self.result[j]['h']
                    txt_cy = self.result[j]['cy']

            if len(addString)>0:
                add['address']  =''.join(addString)
                self.res.update(add)
                break
                                
    def authority(self):
        """
        签发机关
        :return:self.res.update(authority)
        """
        authority = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ##出生年月
            res = re.findall('签发机关.*公安.*', txt)
            res = re.findall('.*公安.*', txt)

            if len(res) > 0:
                authority['authority'] = res[0].replace('签发机关','')
                self.res.update(authority)
                break

    def effecttime(self):
        """
                有效期起始时间 有效期结束时间
                :return:self.res.update(authority)
                """
        effect = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ##出生年月
            res = re.findall('有效期限\d{4}\.\d{2}\.\d{2}\-\d{4}\.\d{2}\.\d{2}', txt)
            res = re.findall('\d{4}\.\d{2}\.\d{2}\-\d{4}\.\d{2}\.\d{2}', txt)

            if len(res) > 0:
                effect['effectiveDate'] = res[0].replace('有效期限', '')[0:10]
                effect['expiryDate'] = res[0].replace('有效期限', '')[11:]
                self.res.update(effect)
                break