# file: settings.py
# author: name <email>
# date: 11-11-2019
'''
This is the deeplearning application settings file.
'''

class HyperParameter_Settings():
    def __init__(self):
        # Network hyperparameter settings settings
        self.INPUT_HEIGHT = 400
        self.INPUT_WIDTH = 400
        self.EPOCHS = 200
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.0001
        self.INPUT_CHANNEL = 1
        self.DROPOUT = 0.5
        self.NUM_OF_CLASSES = 4
        self.POOL_SIZE = 2
        self.KERNEL_SIZE = 3
        self.MAX_POOL_SIZE = 2
        self.STRIDES = 1
        self.FILTERS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.TRAINING_STEPS = 1000
        self.PADDING_SAME = "same"
        self.PADDING_VALID = "valid"
        self.TRAIN_SET = []
        self.VALIDATION_SET = []
        self.TEST_SET = []
        self.eps = 1e-10
        self.RES_FILTERS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
        self.WEIGHT_DECAY = 0.01
        self.BATCHNORM_MOMENTUM = 0.95