# coding: utf-8
import numpy as np
import dogs_vs_cats
import dogs_vs_cats.model
from chainer import Variable
import chainer.links as L


if __name__ == '__main__':
    xp = np
    images, labels = dogs_vs_cats.load_images('./dataset/train/')
    model = dogs_vs_cats.model.VGG16()

    X = Variable(xp.array(
        list(map(L.model.vision.vgg.prepare, images)), dtype=xp.float32))
    pred = model(X[:32])
