import math
import os
import pickle
import sys
import timeit
import warnings

import numpy as np
from sklearn.externals._pilutil import imread
warnings.filterwarnings('ignore')


def integral_image(image, h=26, w=40):
    integral_img = np.zeros((h + 1, w + 1))
    integral_img[1:, 1:] = image.copy()
    for i in range(1, h):
        for j in range(1, w):
            integral_img[i][j] = integral_img[i][j] \
                                 - integral_img[i - 1][j - 1] \
                                 + integral_img[i][j - 1] \
                                 + integral_img[i - 1][j]
    return integral_img[1:, 1:]


def read_file(filename) -> np.array:
    image = np.array(imread(filename))
    assert image.shape == (26, 40, 3)
    image = image[:, :, 0]
    integral_img = integral_image(image)
    assert integral_img.shape == (26, 40)
    images = [image]

    # change images to get more samples
    flipped_image = np.fliplr(image)
    images.append(integral_image(flipped_image))
    flipped_image_tr = np.flipud(image)
    images.append(integral_image(flipped_image_tr))
    return images


def get_s(folder):
    base_n = 0

    y = []
    features = []

    flipped_y = []
    flipped_features = []

    for filename in os.listdir(os.path.join(folder, 'cars')):
        name = os.path.join(folder, 'cars', filename)
        integral_img, *flipped = read_file(name)
        y.append(-1)
        features.append(generate_vj_features(integral_img))
        for fl in flipped:
            flipped_y.append(-1)
            flipped_features.append(generate_vj_features(fl))
        base_n += 1

    for filename in os.listdir(os.path.join(folder, 'faces')):
        name = os.path.join(folder, 'faces', filename)
        integral_img, *flipped = read_file(name)
        y.append(1)
        features.append(generate_vj_features(integral_img))

        for fl in flipped:
            flipped_y.append(1)
            flipped_features.append(generate_vj_features(fl))
        base_n += 1

    y.extend(flipped_y)
    features.extend(flipped_features)
    return base_n, y, features


def generate_vj_features(integral_img):
    def calc_rect(x: int, y: int, w: int, h: int):
        return integral_img[x + w][y + h] + integral_img[x][y] \
               - integral_img[x + w][y] - integral_img[x][y + h]

    max_w, max_h = integral_img.shape
    features = []
    for w in range(5, max_w+1):
        for h in range(5, max_h+1):
            for i in range(0, max_w-w, 10):
                for j in range(0, max_h-h, 10):
                    initial = calc_rect(i, j, w, h)
                    if i + 2*w < max_w:
                        initial_right = calc_rect(i+w, j, w, h)
                        features.append([[initial], [initial_right]])

                    if j + 2*h < max_h:
                        initial_bottom = calc_rect(i, j+h, w, h)
                        features.append([[initial], [initial_bottom]])

                    if i + 3*w < max_w:
                        initial_right_second = calc_rect(i+2*w, j, w, h)
                        features.append(
                            [[initial, initial_right_second], [initial_right]])

                    if i+2*w < max_w and j+2*h < max_h:
                        initial_right_bottom = calc_rect(i+w, j+h, w, h)
                        features.append(
                            [[initial_right, initial_bottom],
                             [initial, initial_right_bottom]]
                        )

    final_features = np.zeros(len(features))
    for i, feature in enumerate(features):
        final_features[i] = sum(feature[0]) - sum(feature[1])
    return final_features


class WeakClassifier:
    def __init__(self, threshold: int, class_: int, i: int = None, error=None):
        self.threshold = threshold
        self.class_ = class_
        self.error = error
        self.i = i

    def predict(self, x):
        return self.class_ if x < self.threshold else -self.class_

    def calculate_error(self, x, y):
        self.error = len([1 for i in range(len(x)) if self.predict(x[i]) != y[i]]) \
               / len(x)
        return self.error


def generate_classifier(features, y, weights, total_pos, total_neg):
    min_threshold = None
    min_error = float('inf')
    min_class = None

    current_pos = 0
    current_neg = 0
    s = sorted(zip(weights, features, y), key=lambda x: x[1])
    for i, (w, value, y_) in enumerate(s):
        err_pos = total_pos - current_pos + current_neg
        err_neg = total_neg - current_neg + current_pos
        error = min(err_pos, err_neg)
        if error < min_error:
            min_error = error
            min_threshold = value
            min_class = 1 if err_pos < err_neg else -1

        if y_ == 1:
            current_pos += w
        else:
            current_neg += w

    return WeakClassifier(min_threshold, min_class, error=min_error)


def find_weak(features, y, weights):
    min_error = np.inf
    min_classifier = None

    total_pos = sum([weights[i] for i in range(len(y)) if y[i] == 1])
    total_neg = sum(weights) - total_pos

    for i in range(min(5000, len(features))):
        classifier = generate_classifier(features[i], y, weights, total_pos, total_neg)
        classifier.i = i
        if classifier.error < min_error:
            min_error = classifier.error
            min_classifier = classifier
    return min_classifier


def learn(y, features, start, base_n):
    n = len(y)
    weights = [1 / n] * n

    hs = []
    alphas = []
    base_accuracy = 0
    t = 0
    while base_accuracy < 0.98:
        h = find_weak(features, y, weights)
        alpha = 0.5 * math.log((1 - h.error) / h.error)
        z = 2 * math.sqrt(h.error * (1 - h.error))
        for i in range(len(weights)):
            weights[i] *= math.exp(-alpha * y[i] * h.predict(features[h.i][i])) / z

        hs.append(h)
        alphas.append(alpha)

        base_accuracy = 0
        accuracy = 0
        for i, y_ in enumerate(y[:base_n]):
            pred = 0
            for a, h_ in zip(alphas, hs):
                pred += a * h_.predict(features[h_.i][i])
            class_ = np.sign(pred)
            if class_ == y_:
                accuracy += 1
            if i == base_n - 1:
                base_accuracy = accuracy
        base_accuracy /= base_n * 1.0
        accuracy /= n * 1.0

        t += 1
        # print(f'[{t}] accuracy: {accuracy:.3f}\t'
        #       f'base_acciracy: {base_accuracy:.3f}\t'
        #       f'time: {timeit.default_timer() - start:.2f}s')
    return hs, alphas


def save_model(hs, alphas, file):
    with open(file, 'wb') as f:
        pickle.dump(list(zip(hs, alphas)), f)


def main(model_file, folder='train'):
    start = timeit.default_timer()

    base_n, y, features = get_s(folder)
    features = np.transpose(features)
    hs, alphas = learn(y, features, start, base_n)
    save_model(hs, alphas, model_file)
    assert timeit.default_timer() - start < 600


folder = sys.argv[-2]
model = sys.argv[-1]
main(model, folder)
