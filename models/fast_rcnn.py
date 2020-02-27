

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
from keras.layers import Input, add, Dense, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, TimeDistributed
from hpsettings import HyperParameter_Settings

##################  R-CNN Model  #######################
# RoI Pooling layer
class RoIPooling(Layer):


    def build(self, input_shape):
        self.shape = input_shape
        super(RoIPooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        ind = K.reshape(inputs[2],(-1,))
        x = K.tf.image.crop_and_resize(inputs[0], inputs[1], ind, self.size)
        return x

    def compute_output_shape(self, input_shape):
        a = input_shape[1][0]
        b = self.size[0]
        c = self.size[1]
        d = input_shape[0][3]
        return (a,b,c,d)

class Faster_RCNN(object):
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

    def build_model(self):
        ## implement build model codes
        pass

    def __init__(self, pool_size, num_rois, **kwargs):
        self.dim_ordering = K.image_dim_ordering()
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.num_channels = 1536

        super(Faster_RCNN, self).__init__(**kwargs)

    def build_model(self):
        feature_map = Input(batch_shape=(None, None, None, self.num_channels))
        rois = Input(batch_shape=(None, 4))
        ind = Input(batch_shape=(None, 1),dtype='int32')
        p1 = RoIPooling()([feature_map, rois, ind])
        flat1 = Flatten()(p1)

        fc1 = Dense(
            units=1024,
            activation="relu",
            name="fc2")(flat1)

        fc1 = BatchNormalization()(fc1)

        output_deltas = Dense(
            units=4 * 200,
            activation="linear",
            kernel_initializer="uniform",
            name="deltas2"
            )(fc1)

        output_scores = Dense(
            units=1 * 200,
            activation="softmax",
            kernel_initializer="uniform",
            name="scores2"
            )(fc1)

        model = Model(inputs=[feature_map, rois, ind],outputs=[output_scores,output_deltas])
        model.summary()
        model.compile(optimizer='rmsprop',
            loss={'deltas2':smoothL1, 'scores2':'categorical_crossentropy'})
