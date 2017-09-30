# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import sys
import pickle
import struct
import sys, cv2

import cv2
import scipy.io as sio

# get files in directory
import glob
import os

#caffe_root = '/home/kong/caffe/'
# 运行模型的prototxt
deployPrototxt = '/home/kong/actioncaffe/caffe-action_recog/models/bvlc_googlenet/deploy.prototxt'
# 相应载入的modelfile
modelFile = '/home/kong/actioncaffe/caffe-action_recog/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
# meanfile 也可以用自己生成的
meanFile = '/home/kong/action-attention/extractfeat/ilsvrc_2012_mean.npy'

hmdb51dir = '/home/kong/dataset/subhmdb3'

gpuID = 0
postfix = 'conv'
layerName = 'inception_5b/output'


# 初始化函数的相关操作
def initilize():
    print 'initilize ... '
    #sys.path.insert(0, caffe_root + 'python')
    caffe.set_mode_gpu()
    # caffe.set_device(gpuID)
    net = caffe.Net(deployPrototxt, modelFile, caffe.TEST)
    return net


# 提取特征并保存为相应地文件
def extractFeature(net):
    # 对输入数据做相应地调整如通道、尺寸等等
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    mu = np.load(meanFile)
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    transformer.set_mean('data', mu)  # mean pixel
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    # set net to batch size of 1 如果图片较多就设置合适的batchsize
    net.blobs['data'].reshape(40, 3, 224, 224)  # 这里根据需要设定，如果网络中不一致，需要调整
    # extract feat for folder
    classdirs = os.listdir(hmdb51dir)
    # 遍历classdirs
    for childdir in classdirs:
        # 针对51个当中的每一个子目录，获取所有avi
        # 在imagesdir根目录下创建子目录
        print('processing ', childdir)
        # os.mkdir(imagesdir+'/'+childdir)
        # opencv capture VideoWriter
        # ensure avi exist
        for file in glob.glob(hmdb51dir + '/' + childdir + '/*.avi'):
            # 针对每一个avi提取image序列 ti qu wan cheng le
            thisdir = file.replace(".avi", "")
            print('extracting ', file)
            # os.mkdir(thisdir)
            myjpgurls = glob.glob(thisdir + '/*.jpg')
            for myjpgurl in myjpgurls:
                # extract feat
                imagefile_abs = myjpgurl
                print imagefile_abs
                net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile_abs))
                out = net.forward()
                fea_file = imagefile_abs.replace('.jpg', postfix)

                # test
                print type(net.blobs[layerName].data)
                print net.blobs[layerName].data.shape
                feat = net.blobs[layerName].data[0]
                print type(feat)
                print feat.shape
                np.save(fea_file, feat)


# 读取文件列表
def readImageList(imageListFile):
    imageList = []
    with open(imageListFile, 'r') as fi:
        while (True):
            line = fi.readline().strip().split()  # every line is a image file name
            if not line:
                break
            imageList.append(line[0])
    print 'read imageList done image num ', len(imageList)
    return imageList


if __name__ == "__main__":
    net = initilize()
    # imageList = readImageList(imageListFile)
    extractFeature(net)