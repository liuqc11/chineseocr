# -*- coding: utf-8 -*-
import os
import logging
#pwd = os.getcwd()
pwd = os.path.abspath(os.path.dirname(__file__))

#######################是否使用GPU######################
## GPU选择及启动GPU序号
GPU = True##OCR 是否启用GPU
GPUID=0##调用GPU序号

# ## nms选择,支持cython,gpu,python
# nmsFlag='gpu'## cython/gpu/python ##容错性 优先启动GPU，其次是cpython 最后是python
# if not GPU:
#     nmsFlag='cython'
#######################是否使用GPU######################


######################文字方向检测######################
##vgg文字方向检测模型
DETECTANGLE=False##是否进行文字方向检测
AngleModelPb = os.path.join(pwd,"models","Angle_Detection","Angle-model.pb")
AngleModelPbtxt = os.path.join(pwd,"models","Angle_Detection","Angle-model.pbtxt")
AngleModelFlag  = 'opencv'  ## opencv or tf
######################文字方向检测######################



########################文字检测########################
##文字检测引擎
yoloTextFlag = 'keras' ##keras,opencv,darknet，模型性能 keras>darknet>opencv
IMGSIZE = (608,608)## yolo3 输入图像尺寸

############## keras yolo  ##############
keras_anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
class_names = ['none','text',]
kerasTextModel=os.path.join(pwd,"models","Text_Detection","text.h5")##keras版本模型权重文件
############## keras yolo  ##############

############## darknet yolo  ##############
darknetRoot = os.path.join(os.path.curdir,"darknet")## yolo 安装目录
yoloCfg     = os.path.join(pwd,"models","Text_Detection","text.cfg")
yoloWeights = os.path.join(pwd,"models","Text_Detection","text.weights")
yoloData    = os.path.join(pwd,"models","Text_Detection","text.data")
############## darknet yolo  ##############
########################文字检测########################


########################OCR模型#########################
ocr_redis = False##是否多任务执行OCR识别加速 如果多任务，则配置redis数据库，数据库账号参考apphelper/redisbase.py

##OCR模型是否调用LSTM层
LSTMFLAG = True ##是否启用LSTM crnn模型
ocrFlag = 'torch' ##ocr模型 支持 keras,torch版本
chineseModel = True##模型选择 True:中英文模型 False:英文模型
ocrModelKeras = os.path.join(pwd,"models","Text_Recognition","ocr-dense-keras.h5")##keras版本OCR，暂时支持dense

# if chinsesModel:
#     if LSTMFLAG:
#         ocrModel  = os.path.join(pwd,"models","ocr-lstm.pth")
#     else:
#         ocrModel = os.path.join(pwd,"models","ocr-dense.pth")
# else:
#         ##纯英文模型
#         ocrModel = os.path.join(pwd,"models","ocr-english.pth")
##转换keras模型 参考tools目录
ocrModelKerasDense       = os.path.join(pwd,"models","Text_Recognition","ocr-dense-keras.h5")
ocrModelKerasLstm        = os.path.join(pwd,"models","Text_Recognition","ocr-lstm-keras.h5")
ocrModelKerasEng         = os.path.join(pwd,"models","Text_Recognition","ocr-english-keras.h5")

ocrModelTorchLstm        = os.path.join(pwd,"models","Text_Recognition","ocr-lstm.pth")
ocrModelTorchDense       = os.path.join(pwd,"models","Text_Recognition","ocr-dense.pth")
ocrModelTorchEng         = os.path.join(pwd,"models","Text_Recognition","ocr-english.pth")

ocrModelOpencv           = os.path.join(pwd,"models","Text_Recognition","ocr.pb")
########################OCR模型#########################

TIMEOUT=30##超时时间

########################Web日志#########################
log_file = "logs/webpy.log" # 日志文件路径 #
logformat = "[%(asctime)s] %(filename)s:%(lineno)d(%(funcName)s): [%(levelname)s] %(message)s" # 日志格式 #
datefmt = "%Y-%m-%d %H:%M:%S" # 日志中显示的时间格式 #
loglevel = logging.INFO
log_interval = "d" # 每隔一天生成一个日志文件#
log_backups = 3 # 后台保留3个日志文件 #
########################Web日志#########################