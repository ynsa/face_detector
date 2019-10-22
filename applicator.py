#!/usr/bin/env python3
import pickle
import warnings

import numpy as np
from sklearn.externals._pilutil import imread
import sys
from learner import generate_vj_features, WeakClassifier
warnings.filterwarnings('ignore')


def read_file(fname):
    image = imread(fname)
    assert image.shape == (26, 40, 3)
    return image[:,:,0]


def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(model, features):
    pred = 0
    for h_, a in model:
        pred += a * h_.predict(features[h_.i])
    return int(np.sign(pred))


def main(fname, model_file='my.model'):
    integral_img = read_file(fname)
    features = generate_vj_features(integral_img)
    model = load_model(model_file)
    class_ = predict(model, features)
    print(int(class_ == 1))


model = sys.argv[-2]
fname = sys.argv[-1]

main(fname, model)
