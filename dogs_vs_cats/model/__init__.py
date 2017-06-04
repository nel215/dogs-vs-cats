# coding: utf-8
import chainer.links as L
from chainer import Chain


class VGG16(Chain):

    def __init__(self):
        super(VGG16, self).__init__(
            vgg16=L.VGG16Layers(),
            fc8=L.Linear(4096, 1),
        )

    def __call__(self, x):
        h1 = self.vgg16(x, layers=['fc7'])['fc7']
        h2 = self.fc8(h1)
        return h2
