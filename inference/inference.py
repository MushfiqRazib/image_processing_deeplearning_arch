import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf

from models import *

from data.train import *
from data.test import *
from settings import Configuration_Settings
import time
from hpsettings import HyperParameter_Settings
from metrics import Metrics
from models.fcn import FullyConvolutionalNetwork_Model
from models.fast_rcnn import Faster_RCNN

class Inference(object):
    def __init__(self):
        self.conf = Configuration_Settings()
        self.hparams = HyperParameter_Settings()
        self.hparams.INPUT_HEIGHT = 400
        self.hparams.INPUT_WIDTH = 400
        self.hparams.INPUT_CHANNEL = 1
        self.hparams.EPOCHS = 200
        self.hparams.BATCH_SIZE = 32
        self.hparams.LEARNING_RATE = 0.001
        self.hparams.NUM_OF_CLASSES = 4
        self.hparams.TRAINING_STEPS = 1000
        self.hparams.WEIGHT_DECAY = 0.01
        self.hparams.BATCHNORM_MOMENTUM = 0.95


    def inference_fcn(self):
        label_dir = self.conf.TRAIN_LABELS_DIR
        data_dir = self.conf.TRAIN_DATA_DIR
        current_dir = os.path.dirname(os.path.realpath(__file__))
        data_suffix='.jpg'
        label_suffix='.png'

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        session = tf.Session(config=config)
        K.set_session(session)

        save_path = os.path.join(current_dir, 'saved_models/' + self.model_name)
        checkpoint_path = os.path.join(save_path, 'fcn_checkpoint_weights.hdf5')
        image_list = data_dir #'2007_000491'
        fcnModel = FullyConvolutionalNetwork_Model()
        model = fcnModel.build_model()
        model.load_weights(checkpoint_path, by_name=True)
        model.summary()

        results = []
        total = 0
        for img_num in image_list:
            img_num = img_num.strip('\n')
            total += 1
            print('#%d: %s' % (total,img_num))
            image = Image.open('%s/%s%s' % (data_dir, img_num, data_suffix))
            image = img_to_array(image)  # , data_format='default')

            label = Image.open('%s/%s%s' % (label_dir, img_num, label_suffix))
            label_size = label.size

            img_h, img_w = image.shape[0:2]

            # long_side = max(img_h, img_w, image_size[0], image_size[1])
            pad_w = max(image_size[1] - img_w, 0)
            pad_h = max(image_size[0] - img_h, 0)
            image = np.lib.pad(image, ((pad_h/2, pad_h - pad_h/2), (pad_w/2, pad_w - pad_w/2), (0, 0)), 'constant', constant_values=0.)
            # image -= mean_value
            '''img = array_to_img(image, 'channels_last', scale=False)
            img.show()
            exit()'''
            # image = cv2.resize(image, image_size)

            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            result = model.predict(image, batch_size=1)
            result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)

            result_img = Image.fromarray(result, mode='P')
            result_img.palette = label.palette
            # result_img = result_img.resize(label_size, resample=Image.BILINEAR)
            result_img = result_img.crop((pad_w/2, pad_h/2, pad_w/2+img_w, pad_h/2+img_h))
            # result_img.show(title='result')
            if return_results:
                results.append(result_img)
            if save_dir:
                result_img.save(os.path.join(save_dir, img_num + '.png'))
        return results

    def show_results_fcn_inference(self, results):
        for result in results:
            result.show(title='result', command=None)
