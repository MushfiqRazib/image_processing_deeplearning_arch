import argparse
import sys
import os

from train_networks import Train_Networks
from inference.inference import Inference

if __name__ == '__main__':
    # main entry

    parser = argparse.ArgumentParser()
    parser.add_argument('-alg', metavar='TYPE', type=str, help='fcn or frcnn')
    parser.add_argument('-traintype', metavar='TYPE', type=str, help='train or test')
    #parser.add_argument('-pre', metavar='TYPE', type=str, help='')
    args = parser.parse_args()

    train_nn = Train_Networks()
    infer_nn = Inference()

    if args.alg == 'fcn':
        if args.traintype == "train":
            train_nn.train_fcn()

        if args.traintype == "test":
            results = infer_nn.inference_fcn()
            infer_nn.show_results_fcn_inference(results)

    if args.alg == 'frcnn':
        if args.traintype == "train":
            train_nn.train_fastrcnn()

        #if args.traintype == "test":
