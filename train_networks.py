import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
#import keras.utils.visualize_util as vis_util

from models import *
#from utils.loss_function import *
#from utils.metrics import *
#from utils.SegDataGenerator import *
import time
from hpsettings import HyperParameter_Settings
from metrics import Metrics
from models.fcn import FullyConvolutionalNetwork_Model
from models.fast_rcnn import Faster_RCNN

from data.train import *
from data.test import *
from settings import Configuration_Settings

class Train_Networks(object):
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

    def train_fcn(self):
        train_file_path = self.conf.TRAINING_DATA_DIR
        val_file_path = self.conf.VALIDATION_DATA_DIR
        label_dir = self.conf.TRAIN_LABELS_DIR
        data_dir = self.conf.TRAIN_DATA_DIR
        data_suffix='.jpg'
        label_suffix='.png'
        model_name = 'fcn_model.h5'
        resume_training = False

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        session = tf.Session(config=config)
        K.set_session(session)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        fcnModel = FullyConvolutionalNetwork_Model()
        save_path = os.path.join(current_dir, 'saved_models/' + self.model_name)
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        # ###################### make model ########################
        checkpoint_path = os.path.join(save_path, 'fcn_checkpoint_weights.hdf5')
        model = fcnModel.build_model()

        # ###################### optimizer ########################
        optimizer = SGD(lr=lr_base, momentum=0.9)
        # optimizer = Adam(lr=lr_base, beta_1 = 0.825, beta_2 = 0.99685)

        loss_fn = softmax_sparse_crossentropy_ignoring_last_label
        metrics = [sparse_accuracy_ignoring_last_label]
        loss_shape = None
        ignore_label = 255
        label_cval = 255

        model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=metrics)
        if resume_training:
            model.load_weights(checkpoint_path, by_name=True)
        model_path = os.path.join(save_path, "fcn_model.json")
        # save model structure
        f = open(model_path, 'w')
        model_json = model.to_json()
        f.write(model_json)
        f.close
        img_path = os.path.join(save_path, "model.png")
        # #vis_util.plot(model, to_file=img_path, show_shapes=True)
        model.summary()
        callbacks = [lr_scheduler]

        # ####################### tfboard ###########################
        if K.backend() == 'tensorflow':
            tensorboard = TensorBoard(log_dir=os.path.join(save_path, 'logs'), histogram_freq=10, write_graph=True)
            callbacks.append(tensorboard)
        # ################### checkpoint saver#######################
        checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'fcn_checkpoint_weights.hdf5'), save_weights_only=True) #.{epoch:d}
        callbacks.append(checkpoint)
        # set data generator and train
        #train_datagen = ## training data generator  from data.train import *
        #val_datagen = ## validation data generator  from data.test import *

        def get_file_len(file_path):
            fp = open(file_path)
            lines = fp.readlines()
            fp.close()
            return len(lines)

        # from Keras documentation: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished
        # and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
        steps_per_epoch = int(np.ceil(get_file_len(train_file_path) / float(batch_size)))

        history = model.fit_generator(generator=train_datagen, validation_data=val_datagen, steps_per_epoch=steps_per_epoch,
                                 epochs=self.hparams.EPOCHS, verbose=1, max_queue_size=10,
                                 workers=num_workers, use_multiprocessing=use_multiprocessing, shuffle=False,
                                 callbacks=callbacks)

        model.save_weights(save_path+'/model.hdf5')

    def train_fastrcnn(self):
        ## fast rcnn training.
        train_file_path = self.conf.TRAINING_DATA_DIR
        val_file_path = self.conf.VALIDATION_DATA_DIR
        label_dir = self.conf.LABELS_DIR
        data_dir = self.conf.DATA_DIR

    def scheduler(epoch):
        if self.hparams.EPOCHS < 50:
            return self.hparams.hparams.LEARNING_RATE
        else:
            return self.hparams.hparams.LEARNING_RATE * tf.math.exp(0.1 * (10 - epoch))
