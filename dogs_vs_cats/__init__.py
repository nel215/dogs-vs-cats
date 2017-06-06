# coding: utf-8
import os
import numpy as np
import cupy
import chainer
import chainer.links as L
import chainer.functions as F
import dogs_vs_cats.model
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from chainer import Variable
from chainer.optimizers import SGD


def load_images(img_dir, trial):
    '''
    Returns:
        - images: Array of PIL.image
        - labels: Array of int
    '''
    images, labels = [], []
    files = np.random.permutation(os.listdir(img_dir))
    if trial:
        files = files[:48 * 3]
    for fname in files:
        label = fname.split('.')[0]
        fpath = os.path.join(img_dir, fname)
        img = Image.open(fpath)
        images.append(img.copy())
        labels.append(binarize_label(label))
        img.close()

    return images, labels


def binarize_label(label):
    return 1 if label == 'dog' else 0


def train_vgg16(gpu, trial):
    def train(X_train, X_test, y_train, y_test):
        n_train = len(X_train)
        n_test = len(X_test)

        if gpu is not None:
            xp = cupy
            model = dogs_vs_cats.model.VGG16().to_gpu()
        else:
            xp = np
            model = dogs_vs_cats.model.VGG16()

        optimizer = SGD(lr=0.001)
        optimizer.setup(model)

        batch_size = 48
        for epoch in range(100):
            chainer.using_config('train', True)
            perm = np.random.permutation(n_train)
            train_loss = []
            for start in range(0, n_train, batch_size):
                end = min(n_train, start + batch_size)
                model.cleargrads()
                X = Variable(xp.array(X_train[perm[start:end]]))
                y = xp.array(y_train[perm[start:end]]).reshape((-1, 1))
                pred = model(X)
                loss = F.sigmoid_cross_entropy(pred, y)
                train_loss.append(float(loss.data.copy()))
                loss.backward()
                optimizer.update()
            print("train loss:", np.mean(train_loss))

            # validation
            chainer.using_config('train', False)
            pred = xp.zeros((n_test, 1), dtype=np.float32)
            for start in range(0, n_test, batch_size):
                end = min(n_test, start + batch_size)
                pred[start:end] = model(xp.array(X_test[start:end])).data
            loss = F.sigmoid_cross_entropy(
                pred, xp.array(y_test).reshape((-1, 1)))
            print("test loss:", loss.data)

    images, labels = dogs_vs_cats.load_images('./dataset/train/', trial)
    labels = np.array(labels, dtype=np.int32)

    images = np.array(
        list(map(L.model.vision.vgg.prepare, images)), dtype=np.float32)

    skf = StratifiedKFold(n_splits=3, random_state=215)
    for train_idx, test_idx in skf.split(images, labels):
        X_train, X_test = images[train_idx], images[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        train(X_train, X_test, y_train, y_test)
