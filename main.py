# coding: utf-8
import numpy as np
import dogs_vs_cats
import dogs_vs_cats.model
import argparse
import cupy
from chainer import Variable
import chainer.links as L


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the id of gpu', type=int)
    args = parser.parse_args()

    if args.gpu is not None:
        xp = cupy
        model = dogs_vs_cats.model.VGG16().to_gpu()
    else:
        xp = np
        model = dogs_vs_cats.model.VGG16()

    images, labels = dogs_vs_cats.load_images('./dataset/train/')

    X = Variable(xp.array(
        list(map(L.model.vision.vgg.prepare, images)), dtype=xp.float32))
    pred = model(X[:32])
