# -*- coding: utf-8 -*-

""" iMaterialist Challenge at FGVC 2017

Links:
    [iMaterialist Challenge at FGVC 2017](https://www.kaggle.com/c/imaterialist-challenge-FGVC2017)
"""


from __future__ import absolute_import, division, print_function

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import cv2
import numpy as np


data_dir = '/home/khan/workspace/ml_ws/datasets/imat_dataset/'
model_path = "model/"

# load data
# train_X = np.load('data/train_X-100-100-3.npy')
# train_X = train_X.astype(np.float64)
val_X = np.load('data/val_X-100-100-3.npy')
val_X = val_X.astype(np.float64)
# test_X = np.load('data/test_X-100-100-3.npy')
# test_X = test_X.astype(np.float64)
# train_Y = np.load('data/train_Y.npy')
val_Y = np.load('data/val_Y.npy')
# val_Y = val_Y[:,0:2]

CLASS_COUNT = val_Y.shape[1]
print('CLASS_COUNT:\t',CLASS_COUNT)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=5.0)

n = 5

net = input_data(shape=[None, 100, 100, 3])

# Building Residual Network
# net = tflearn.input_data(shape=[None, 100, 100, 3],
#                          data_preprocessing=img_prep,
#                          data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.resnext_block(net, n, 16, 32)
net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 32, 32)
net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 64, 32)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, CLASS_COUNT, activation='softmax')
opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=opt,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='model_resnext_imat',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

model.fit(val_X, val_Y, n_epoch=5, validation_set=0.1,
          snapshot_epoch=True, snapshot_step=100,
          show_metric=True, batch_size=128, shuffle=True,
          run_id='resnext_imat')

model.save('models/imat.tflearn')
