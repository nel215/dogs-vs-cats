# coding: utf-8
import dogs_vs_cats
import numpy as np
import chainer
import chainer.links as L
from chainer import Chain, Variable


class VGG16(Chain):

    def __init__(self):
        super(VGG16, self).__init__(
            vgg16=L.VGG16Layers(),
            fc8=L.Linear(4096, 2),
        )

    def __call__(self, x):
        h1 = self.vgg16(x, layers=['fc7'])['fc7']
        h2 = self.fc8(h1)
        return h2


if __name__ == '__main__':
    xp = np
    images, labels = dogs_vs_cats.load_images('./dataset/train/')
    model = VGG16()

    X = Variable(xp.array(
        list(map(L.model.vision.vgg.prepare, images)), dtype=xp.float32))
    pred = model(X[:32])
