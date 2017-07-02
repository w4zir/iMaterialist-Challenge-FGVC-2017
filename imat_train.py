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
train_X = np.load('data/train_X-100-100-3.npy')
train_X = train_X.astype(np.float64)
train_Y = np.load('data/train_Y.npy')

val_X = np.load('data/val_X-100-100-3.npy')
val_X = val_X.astype(np.float64)
val_Y = np.load('data/val_Y.npy')

# test_X = np.load('data/test_X-100-100-3.npy')
# test_X = test_X.astype(np.float64)


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

# Convolutional network building
network = input_data(shape=[None, 100, 100, 3],
                        data_preprocessing=img_prep,
                        data_augmentation=img_aug)

# network = input_data(shape=[None, 100, 100, 3])

network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = fully_connected(network, 256, activation='relu', regularizer="L2")
network = dropout(network, 0.5)
network = fully_connected(network, 256, activation='relu', regularizer="L2")
network = dropout(network, 0.5)
network = fully_connected(network, CLASS_COUNT, activation='softmax')
network = regression(network, optimizer='adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)

# Define model object
model = tflearn.DNN(network, tensorboard_verbose=0)

# Load Model into model object
# model.load('models/driver.tflearn')

# Start finetuning
model.fit(train_X, train_Y, n_epoch=100, validation_set=(val_X,val_Y), shuffle=True,
          show_metric=True, batch_size=64, snapshot_epoch=True,
          snapshot_step=500, run_id='imat_challenge')

model.save('models/imat.tflearn')
