# coding: utf-8
import numpy as np
import dogs_vs_cats
import dogs_vs_cats.model
import argparse
import cupy
from sklearn.preprocessing import LabelBinarizer
from chainer import Variable
import chainer.links as L
import chainer.functions as F


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
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    labels = labels.astype(np.int32)

    images = np.array(
        list(map(L.model.vision.vgg.prepare, images)), dtype=np.float32)
    n = len(images)
    perm = np.random.permutation(n)[:48]
    X = Variable(xp.array(images[perm]))
    y = Variable(xp.array(labels[perm]))
    pred = model(X)
    loss = F.sigmoid_cross_entropy(pred, y)
    loss.backward()
