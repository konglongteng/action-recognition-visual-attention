# -*- coding: utf-8 -*-
"""
This example shows how to read and write float datatypes to a dataset.  The
program first writes floats to a dataset with a dataspace of DIM0xDIM1, then
closes the file.  Next, it reopens the file, reads back the data, and outputs
it to the screen.
此程序读取目录下的numpy文件
循环写入HDFdataset，train valid test hmdb51将近600G
"""
import numpy as np
import h5py
import os
import glob

datapath = "/home/kong/dataset/subhmdb3"
trainfilepath ="/home/kong/dataset/subhmdb3TrainTestsplit"
FILE = "valid_features.h5"
file_framenum = "train_framenum.txt"
file_labels = "train_labels.txt"
file_filename = "train_filename.txt"

#
# FILE = "valid_features.h5"
# file_framenum = "valid_framenum.txt"
# file_labels = "valid_labels.txt"
# file_filename = "valid_filename.txt"

# FILE_valid = "valid_features.h5"
# file_valid_framenum = "valid_framenum.txt"
# file_valid_labels = "valid_labels.txt"
# file_valid_filename = "valid_filename.txt"
#
# FILE_test = "test_features.h5"
# file_test_framenum = "train_framenum.txt"
# file_test_labels = "train_labels.txt"
# file_test_filename = "train_filename.txt"

DATASET = "features"
traintestflag = '1' #1 是train 2是test



def run():
    # Initialize the data.
    # wdata = np.zeros((DIM0, DIM1), dtype=np.float64)
    # for i in range(DIM0):
    #     for j in range(DIM1):
    #         wdata[i][j] = i / (j + 0.5) + j
    DIM0 = 0
    DIM1 = 7*7*1024
    # 计算DIM0 有多少维度
    classdirs = os.listdir(datapath)
    # 遍历classdirs
    for childdir in classdirs:
        # 针对51个当中的每一个子目录，获取所有avi
        # 在imagesdir根目录下创建子目录
        print('processing ', childdir)
        # os.mkdir(imagesdir+'/'+childdir)
        # opencv capture VideoWriter
        # Use the test_split file
        f = open(trainfilepath + '/' + childdir + '_test_split1.txt', "r")
        lines = f.readlines()  # 读取全部内容
        for line in lines:
            linearr = line.split()
            if linearr[1] == traintestflag:
                nameofavi = linearr[0]
                thisdir = nameofavi.replace(".avi", "")
                thisdir = datapath + '/' + childdir + '/' + thisdir
                print('extracting ', nameofavi)
                mynpyurls = glob.glob(thisdir + '/*.npy')
                frameNO = len(mynpyurls)
                # print mynpys
                DIM0 = DIM0 + frameNO

    print 'dim0 is',DIM0


    #end for 所有的子目录
    #写#frames The train_framenum.txt file contains #frames for each video:
    # 89
    # 123
    # 22
    # 136
    file4no = open(file_framenum, 'w')
    file4labels = open(file_labels, 'w')
    file4name = open(file_filename, 'w')

    uni_label = 0
    hdfDim1Index = 0
    with h5py.File(FILE, 'w') as f:
        dset = f.create_dataset(DATASET, (DIM0, DIM1), dtype=np.float64)
        # dset[...] = wdata

        for childdir in classdirs:
            # 针对51个当中的每一个子目录，获取所有avi
            # 在imagesdir根目录下创建子目录
            print('processing ', childdir)
            # os.mkdir(imagesdir+'/'+childdir)
            # opencv capture VideoWriter
            f = open(trainfilepath + '/' + childdir + '_test_split1.txt', "r")
            lines = f.readlines()  # 读取全部内容
            for line in lines:
                linearr = line.split()
                if linearr[1] == traintestflag:
                    #如果是训练，这里获取该路径的avi名称
                    nameofavi = linearr[0]
                    thisdir = nameofavi.replace(".avi", "")
                    thisdir = datapath + '/' + childdir + '/' +thisdir
                    print('extracting ', nameofavi)
                    mynpyurls = glob.glob(thisdir + '/*.npy')
                    mynpyurls = sorted(mynpyurls)
                    frameNO = len(mynpyurls)
                    file4no.writelines(str(frameNO) + '\n')  # 写入帧数目
                    file4labels.writelines(str(uni_label) + '\n')  # 写入lable
                    avifilename = nameofavi
                    avifilename = childdir + '/' + avifilename
                    file4name.writelines(avifilename + '\n')  # 写入文件名路径
                    # 写入HDF5文件
                    for mynpyurl in mynpyurls:
                        # extract feat
                        print mynpyurl  # 保证写的顺序是对的
                        myconv8x8 = np.load(mynpyurl)
                        print 'loading npy:', mynpyurl,len(myconv8x8)
                        myconv1d = myconv8x8.reshape(1, 7 * 7 * 1024)
                        myconv1d = np.squeeze(myconv1d)
                        dset[hdfDim1Index, ...] = myconv1d
                        hdfDim1Index = hdfDim1Index + 1


            uni_label = uni_label + 1

        file4no.close()
        file4labels.close()
        file4name.close()
        print 'total dim1',hdfDim1Index


    # with h5py.File(FILE) as f:
    #     dset = f[DATASET]
    #     rdata = dset[...]

      #print(rdata)


if __name__ == "__main__":
    run()