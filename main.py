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

    if args.gpu is not None:
        xp = cupy
        model = dogs_vs_cats.model.VGG16().to_gpu()
    else:
        xp = np
        model = dogs_vs_cats.model.VGG16()

    optimizer = SGD(lr=0.001)
    optimizer.setup(model)

    images, labels = dogs_vs_cats.load_images('./dataset/train/')
    label_binarizer = LabelBinarizer()
    labels = label_binarizer.fit_transform(labels)
    labels = labels.astype(np.int32)

    images = np.array(
        list(map(L.model.vision.vgg.prepare, images)), dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.05, random_state=215)
    n = len(X_train)

    for epoch in range(100):
        chainer.using_config('train', True)
        perm = np.random.permutation(n)[:48]
        X = Variable(xp.array(X_train[perm]))
        y = Variable(xp.array(y_train[perm]))
        model.cleargrads()
        pred = model(X)
        loss = F.sigmoid_cross_entropy(pred, y)
        print("train loss:", loss.data)
        loss.backward()
        optimizer.update()

        # validation
        chainer.using_config('train', False)
        X = Variable(xp.array(X_test))
        y = Variable(xp.array(y_test))
        pred = model(X)
        loss = F.sigmoid_cross_entropy(pred, y)
        print("test loss:", loss.data)
