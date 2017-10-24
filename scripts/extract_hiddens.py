import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pkl
import numpy
import numpy as np
import scipy

import cv2
import skimage
import skimage.transform
import skimage.io
from PIL import Image


import sys
sys.path.append('../')
import glob

from util.data_handler import DataHandler
from util.data_handler import TrainProto
from util.data_handler import TestTrainProto
#from util.data_handler import TestValidProto
from util.data_handler import TestTestProto
import os
import src.actrec
def overlay(bg,fg):
    """
    Overlay attention over the video frame
    """
    src_rgb = fg[..., :3].astype(numpy.float32) / 255.
    src_alpha = fg[..., 3].astype(numpy.float32) / 255.
    dst_rgb = bg[..., :3].astype(numpy.float32) / 255.
    dst_alpha = bg[..., 3].astype(numpy.float32) / 255.

    out_alpha = src_alpha + dst_alpha * (1. - src_alpha)
    out_rgb = (src_rgb * src_alpha[..., None] + dst_rgb * dst_alpha[..., None] * (1. - src_alpha[..., None])) / out_alpha[..., None]

    out = numpy.zeros_like(bg)
    out[..., :3] = out_rgb * 255
    out[..., 3] = out_alpha * 255

    return out

model ='model_volly.npz'
volleyballdir = '/home/kong/dataset/volleyball-extra'
dataset ='volleyball'

with open('%s.pkl'%model, 'rb') as f:
    options = pkl.load(f)


print 'reading params and building graph by kong...'
params  = src.actrec.init_params(options)
params  = src.actrec.load_params(model, params)
tparams = src.actrec.init_tparams(params)

trng, use_noise, inps, alphas, cost, opt_outs, preds = src.actrec.build_model(tparams, options)
f_alpha = theano.function(inps,alphas,name='f_alpha',on_unused_input='ignore')
f_preds = theano.function(inps,preds,profile=False,on_unused_input='ignore')

# f_h = theano.function(inps,preds,profile=False,on_unused_input='ignore')
f_opt_outs = theano.function(inps,opt_outs,profile=False,on_unused_input='ignore') #opt_outs['hidden'] is the hidden


batch_size = 1
maxlen = options['maxlen']
# try out different fps for visualization
# options['fps'] = 6
fps = options['fps']
skip = int(30/fps)
#data_pb = TestTestProto(batch_size,maxlen,maxlen,dataset,fps) # or TestTrainProto or TestValidProto
data_pb = TestTrainProto(batch_size,maxlen,maxlen,dataset,fps)
dh = DataHandler(data_pb)
dataset_size = dh.GetDatasetSize()
num_batches = dataset_size / batch_size

print 'Data handler ready'
print '-----'

#extracting hiddens for every 10 frames
for tbidx in range(0,dataset_size):

    mask = numpy.ones((maxlen, batch_size)).astype('float32')

    x, y, fname = dh.GetSingleExample(data_pb,tbidx)
    alpha = f_alpha(x,mask,y)
    opt_outs = f_opt_outs(x,mask,y)
    hidden = opt_outs['hidden'] # hidden of lstm_cond_layer 512*1
    sel = opt_outs['selector'] # scalar 10*1  10timestep

    #vidcap = cv2.VideoCapture(videopath+fname) video not vailable
    space = 255.0*numpy.ones((224*2,20,4))
    space[:,:,0:3] = 255.0*numpy.ones((224*2,20,3))

    imgf = numpy.array([]).reshape(2*224,0,4)
    #for ii in xrange(alpha.shape[0]):
        # read frame
        #success, image = vidcap.read()
        # if success:
        #     oldimage = image
        # if not success:
        #     image = oldimage
        # # skip frames according to variable skip
        # for sk in xrange(skip-1):
        #     suc, im = vidcap.read()
        #     if not suc:
        #         break
    globpath = os.path.join(volleyballdir,fname,'*pool.npy')
    myjpgurls = glob.glob(globpath)
    myjpgurls = sorted(myjpgurls)
    print fname
    for ii in xrange(alpha.shape[0]):
        # add an Alpha layer to the RGB image
        poolnpyNmae = myjpgurls[ii]
        cnn = np.load(poolnpyNmae)
        cnn = cnn*sel[ii]
        # concatenate these two features
        cnn1d = np.squeeze(cnn)
        hidden1d = np.squeeze(hidden[ii])
        confeat = np.append(cnn1d, hidden1d)
        confeatName = poolnpyNmae.replace('pool','cc')
        np.save(confeatName,confeat)






# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.figure(figsize=(40,10))
# plt.axis('off')
# dispCount = 10                         # displays the
# width = 224 + (20+224)*(dispCount-1)  # first 6 frames
# ax = plt.imshow(imgf[:,:width,:]/255)
# print '---finish---'