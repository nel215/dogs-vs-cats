# coding: utf-8
import numpy as np
import dogs_vs_cats
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the id of gpu', type=int)
    parser.add_argument('--trial', help='the flag of trial run',
                        action='store_true')
    args = parser.parse_args()

    n_epoch = 20
    vgg_result = dogs_vs_cats.train_vgg16(n_epoch, args.gpu, args.trial)
