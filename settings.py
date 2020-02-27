# file: settings.py
# author: name <email>
# date: 11-11-2019
'''
This is the deeplearning application settings file.
'''

class Configuration_Settings(object):
    def __init__(self):
        # directory settings
        self.LOGS_DIR = "/deeplearning/logs/"
        self.TRAINING_DATA_DIR = "/deeplearning/data/train/"
        self.VALIDATION_DATA_DIR = "/deeplearning/data/val/"
        self.TESTING_DATA_DIR = "/deeplearning/data/test/"
        self.MODELS_SAVING_DIR = "/deeplearning/saved_models/"
        self.TRAIN_LABELS_DIR = "/deeplearning/data/train/annotations/"
        self.TRAIN_DATA_DIR = "/deeplearning/data/train/images/"
        self.TEST_LABELS_DIR = "/deeplearning/data/test/annotations/"
        self.TEST_DATA_DIR = "/deeplearning/data/test/images/"

        # project root settings
        self.ROOT_PATH = r"/home/mushfiqrahman/dev/ProjectRoot"
