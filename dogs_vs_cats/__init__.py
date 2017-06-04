# coding: utf-8
import os
import numpy as np
import cupy
import chainer
import chainer.links as L
import chainer.functions as F
import dogs_vs_cats.model
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from chainer import Variable
from chainer.optimizers import SGD


def load_images(img_dir):
    '''
    Returns:
        - images: Array of PIL.image
        - labels: Array of string
    '''
    images, labels = [], []
    for fname in np.random.permutation(os.listdir(img_dir))[:1000]:
        label = fname.split('.')[0]
        fpath = os.path.join(img_dir, fname)
        img = Image.open(fpath)
        images.append(img)
        labels.append(label)

    return images, labels


def train_vgg16(gpu):
    if gpu is not None:
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
