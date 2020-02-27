from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import traceback
import numpy as np
import numpy.random as npr
import tensorflow as tf
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import Input, Layer
from keras.applications import InceptionResNetV2
from keras.preprocessing.image import load_img, img_to_array
from keras.regularizers import l2
from keras.layers import Input, add, Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, SeparableConv2D, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, TimeDistributed
from hpsettings import HyperParameter_Settings
#from BilinearUpSampling import *

class FullyConvolutionalNetwork_Model():
    def __init__(self):
        self.hparams = deeplearning.hpsettings.HyperParameter_Settings()
        self.hparams.INPUT_HEIGHT = 400
        self.hparams.INPUT_WIDTH = 400
        self.hparams.INPUT_CHANNEL = 1
        self.hparams.KERNEL_SIZE = 3
        self.hparams.EPOCHS = 200
        self.hparams.BATCH_SIZE = 32
        self.hparams.hparams.LEARNING_RATE = 0.0001
        self.hparams.DROPOUT = 0.5
        self.hparams.NUM_OF_CLASSES = 4
        self.hparams.POOL_SIZE = 2
        self.hparams.KERNEL_SIZE = 3
        self.hparams.MAX_POOL_SIZE = 2
        self.hparams.FILTERS = [64, 128, 256, 512, 1024, 2048, 4096]
        self.hparams.TRAINING_STEPS = 1000
        self.hparams.PADDING_SAME = "same"
        self.hparams.PADDING_VALID = "valid"
        self.hparams.RES_FILTERS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.hparams.WEIGHT_DECAY = 0.01
        self.IMAGE_ORDERING = 'channels_last'

    #def build_model(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    def build_model(self):
        img_input = Input(shape=(self.hparams.INPUT_HEIGHT, self.hparams.INPUT_WIDTH, self.hparams.INPUT_CHANNEL))

        # Block 1
        x = Conv2D(self.hparams.FILTERS[0], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block1_conv1', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(img_input)
        x = Conv2D(self.hparams.FILTERS[0], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block1_conv2', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = MaxPooling2D((self.hparams.MAX_POOL_SIZE, self.hparams.MAX_POOL_SIZE), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(self.hparams.FILTERS[1], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block2_conv1', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = Conv2D(self.hparams.FILTERS[1], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block2_conv2', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = MaxPooling2D((self.hparams.MAX_POOL_SIZE, self.hparams.MAX_POOL_SIZE), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(self.hparams.FILTERS[2], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block3_conv1', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = Conv2D(self.hparams.FILTERS[2], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block3_conv2', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = Conv2D(self.hparams.FILTERS[2], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block3_conv3', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = MaxPooling2D((2, 2), strides=(self.hparams.MAX_POOL_SIZE, self.hparams.MAX_POOL_SIZE), name='block3_pool')(x)

        # Block 4
        x = Conv2D(self.hparams.FILTERS[3], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block4_conv1', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = Conv2D(self.hparams.FILTERS[3], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block4_conv2', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = Conv2D(self.hparams.FILTERS[3], (self.hparams.KERNEL_SIZE, self.hparams.KERNEL_SIZE), strides=(1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='block4_conv3', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = MaxPooling2D((2, 2), strides=(self.hparams.MAX_POOL_SIZE, self.hparams.MAX_POOL_SIZE), name='block4_pool')(x)

        # Block 5
        x = Conv2D(self.hparams.FILTERS[3], (3, 3), activation='relu', padding=self.hparams.PADDING_SAME, name='block5_conv1', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = Conv2D(self.hparams.FILTERS[3], (3, 3), activation='relu', padding=self.hparams.PADDING_SAME, name='block5_conv2', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = Conv2D(self.hparams.FILTERS[3], (3, 3), activation='relu', padding=self.hparams.PADDING_SAME, name='block5_conv3', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = MaxPooling2D((2, 2), strides=(self.hparams.MAX_POOL_SIZE, self.hparams.MAX_POOL_SIZE), name='block5_pool')(x)

        # Convolutional layers transfered from fully-connected layers
        x = Conv2D(self.hparams.FILTERS[6], (7, 7), activation='relu', padding=self.hparams.PADDING_SAME, name='fc1', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = Dropout(self.hparams.DROPOUT)(x)
        x = Conv2D(self.hparams.FILTERS[6], (1, 1), activation='relu', padding=self.hparams.PADDING_SAME, name='fc2', kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = Dropout(self.hparams.DROPOUT)(x)
        #classifying layer
        x = Conv2D(self.hparams.NUM_OF_CLASSES, (1, 1), kernel_initializer='he_normal', activation='linear', padding=self.hparams.PADDING_VALID, strides=(1, 1), data_format=self.IMAGE_ORDERING, kernel_regularizer=l2(self.hparams.WEIGHT_DECAY))(x)
        x = BilinearUpSampling2D(size=(32, 32))(x)
        model = Model(img_input, x)

        weights_path = os.path.expanduser(os.path.join('~', '.deeplearning/saved_models/fcn_model.h5'))
        model.load_weights(weights_path, by_name=True)
        return model
