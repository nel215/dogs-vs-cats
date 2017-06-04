# coding: utf-8
import numpy as np
import dogs_vs_cats
import dogs_vs_cats.model
import argparse
import cupy
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from chainer import Variable
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.optimizers import SGD


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the id of gpu', type=int)
    args = parser.parse_args()

    dogs_vs_cats.train_vgg16(args.gpu)
