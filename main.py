# coding: utf-8
import numpy as np
import dogs_vs_cats
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the id of gpu', type=int)
    args = parser.parse_args()

    dogs_vs_cats.train_vgg16(args.gpu)
