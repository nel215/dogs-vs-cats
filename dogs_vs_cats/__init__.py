# coding: utf-8
import os
import numpy as np
from PIL import Image


def load_images(img_dir):
    '''
    Returns:
        - images: Array of PIL.image
        - labels: Array of string
    '''
    images, labels = [], []
    for fname in np.random.permutation(os.listdir(img_dir))[:100]:
        label = fname.split('.')[0]
        fpath = os.path.join(img_dir, fname)
        img = Image.open(fpath)
        images.append(img)
        labels.append(label)

    return images, labels
