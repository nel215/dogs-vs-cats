# coding: utf-8
import os
import numpy as np
import pandas as pd
import cupy as cp
import chainer
import chainer.links as L
import chainer.functions as F
import dogs_vs_cats.model
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from chainer import Variable
from chainer.optimizers import SGD


def load_metadata(img_dir, trial):
    '''
    Returns: metadata: pandas.DataFrame
    '''
    metadata = []
    files = np.random.permutation(os.listdir(img_dir))
    if trial:
        files = files[:48 * 3]
    for fname in files:
        label = fname.split('.')[0]
        fpath = os.path.join(img_dir, fname)
        metadata.append([binarize_label(label), fname, fpath])

    metadata = pd.DataFrame(metadata,
                            columns=['label', 'filename', 'filepath'])

    return metadata


def load_images(filepaths):
    '''
    Returns: images: Array of PIL.image
    '''
    images = []
    for fpath in filepaths:
        img = Image.open(fpath)
        images.append(img.copy())
        img.close()

    return images


def binarize_label(label):
    return 1 if label == 'dog' else 0


class PredictionTask(object):
    def __init__(self, gpu, batch_size):
        self.xp = cp if gpu is not None else np
        self.gpu = gpu
        self.batch_size = batch_size

    def _create_model(self):
        if self.gpu is not None:
            return dogs_vs_cats.model.VGG16().to_gpu()
        else:
            return dogs_vs_cats.model.VGG16()

    def _train_once(self, model, optimizer, X_train, y_train):
        batch_size = self.batch_size
        xp = self.xp
        n_train = len(X_train)
        perm = np.random.permutation(n_train)
        train_loss = []
        for start in range(0, n_train, batch_size):
            end = min(n_train, start + batch_size)
            model.cleargrads()
            images = load_images(X_train[perm[start:end]])
            images = list(map(L.model.vision.vgg.prepare, images))
            X = Variable(xp.array(images, dtype=np.float32))
            y = xp.array(y_train[perm[start:end]]).reshape((-1, 1))
            pred = model(X)
            loss = F.sigmoid_cross_entropy(pred, y)
            train_loss.append(float(loss.data.copy()))
            loss.backward()
            optimizer.update()
        print("train loss:", np.mean(train_loss))

    def _predict(self, model, X_test):
        batch_size = self.batch_size
        xp = self.xp
        n_test = len(X_test)
        pred = xp.zeros((n_test, 1), dtype=np.float32)
        for start in range(0, n_test, batch_size):
            end = min(n_test, start + batch_size)
            images = load_images(X_test[start:end])
            X = xp.array(list(map(
                L.model.vision.vgg.prepare, images)), dtype=np.float32)
            pred[start:end] = model(X).data
        return pred

    def run(self, n_epoch, trial):
        def train(X_train, X_test, y_train, y_test):
            xp = self.xp

            model = self._create_model()

            optimizer = SGD(lr=0.001)
            optimizer.setup(model)

            y_test = xp.array(y_test).reshape((-1, 1))
            for epoch in range(n_epoch):
                print('epoch:', epoch)
                chainer.using_config('train', True)
                self._train_once(
                    model, optimizer, X_train, y_train)

                with chainer.no_backprop_mode():
                    chainer.using_config('train', False)
                    pred = self._predict(model, X_test)
                    loss = F.sigmoid_cross_entropy(pred, y_test)
                    print("test loss:", loss.data)

            if self.gpu is not None:
                pred = chainer.cuda.to_cpu(pred)
            return pred

        metadata = load_metadata('./dataset/train/', trial)
        filepaths = np.array(metadata['filepath'])
        labels = np.array(metadata['label'], dtype=np.int32)

        cv_pred = np.zeros(len(labels))
        skf = StratifiedKFold(n_splits=3, random_state=215)
        for train_idx, test_idx in skf.split(filepaths, labels):
            X_train, X_test = filepaths[train_idx], filepaths[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            pred = train(X_train, X_test, y_train, y_test)
            cv_pred[test_idx] = pred.reshape(len(pred))
        metadata['vgg_pred'] = cv_pred

        return metadata
