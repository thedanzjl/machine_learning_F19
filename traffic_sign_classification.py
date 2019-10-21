import matplotlib.pyplot as plt
import csv
import cv2
import random
import numpy as np
from math import floor, ceil
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from time import time

frequency = [0 for _ in range(0, 43)]


class Augmented:

    @staticmethod
    def add_light(image, gamma=None):
        if gamma is None:
            gamma = random.random() * 4
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        im = cv2.LUT(image, table)
        return im

    @staticmethod
    def contrast(image, contrast=None):
        if contrast is None:
            contrast = random.random() * 20
        im = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        im[:, :, 2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for
                       row in im[:, :, 2]]
        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
        return im


def read_traffic_signs(path):
    """
    Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    :param path: path to the traffic sign data, for example './GTSRB/Training
    :returns: list of images, list of corresponding labels
    """
    images = list()  # images
    labels = list()  # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        prefix = path + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        gtReader.__next__()  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))  # the 1th column is the filename
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels


def transform_to_same_size(images, shape=(30, 30)):
    result = list()
    for img in images:
        height, width = img.shape[0], img.shape[1]
        if height > width:  # pad left & right
            top = bottom = 0
            left = floor((height - width) / 2)
            right = ceil((height - width) / 2)
        else:
            left = right = 0
            top = floor((width - height) / 2)
            bottom = ceil((width - height) / 2)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # img is rgb, convert to opencv's default bgr
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)  # pad
        img = cv2.resize(img, shape)  # resize
        result.append(img)
    return result


def split_and_frequency(images, labels):
    """
    Splits images to training and validation sets in proportion 80%-20% relatively
    """
    global frequency
    training = list()
    validation = list()
    training_labels = list()
    validation_labels = list()

    count = 0
    track_goes_to_training = True
    for i in range(len(images)):
        if count == 30:
            count = 0
        if count == 0:
            track_goes_to_training = random.random() < 0.8
            if track_goes_to_training:
                frequency[int(labels[i])] += 30

        if track_goes_to_training:
            training.append(images[i])
            training_labels.append(labels[i])
        else:
            validation.append(images[i])
            validation_labels.append(labels[i])
        count += 1

    plt.bar(range(0, 43), frequency)
    plt.ylabel('Samples')
    plt.xlabel('Class')

    plt.show()

    return training, training_labels, validation, validation_labels


def augmentation(training, training_labels):
    max_frequency = max(frequency)
    for cls in range(0, 43):
        n_images = frequency[cls]
        start = sum(frequency[:cls])
        end = start + n_images
        i = start
        while n_images < max_frequency:
            img = training[i]
            aug = Augmented.add_light(img) if random.random() <= 0.5 else Augmented.contrast(img)
            training.append(aug)
            training_labels.append(cls)
            i += 1
            if i == end:
                i = start
            n_images += 1
    # return training, training_labels


def shuffle(images, labels):
    shuffled = list(zip(images, labels))
    random.shuffle(shuffled)
    imgs = [i[0] for i in shuffled]
    lbls = [i[1] for i in shuffled]
    return imgs, lbls


def normalize(images):
    for i in range(len(images)):
        images[i] = images[i].flatten()
        images[i] = images[i] / 255.0


def preprocess_and_train(X, y, X_test, y_test, shape=(30, 30)):
    """
    Transforms images to the same shape, trains the model and returns (accuracy of the model, time to train the model)
    """
    X = transform_to_same_size(X, shape=shape)
    X_test = transform_to_same_size(X_test, shape=shape)

    print('normalizing images...')
    normalize(X)
    normalize(X_test)

    print('training the model...')
    t = time()  # start timer
    clf = RandomForestClassifier(n_estimators=30, max_depth=50)
    clf.fit(X, y)
    t = time() - t  # calculate how much time it took to train the model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, t


def experiment_and_analyze(X, y, X_test, y_test, augment=True, title=''):
    """
    makes transformation of images 5 times with different shapes and trains. Eventually it plots the dependence
    of time on shape and accuracy on shape
    """
    shapes = [(20, 20), (30, 30), (50, 50), (70, 70), (100, 100)]
    accuracies = list()
    times = list()

    if augment:
        print('data augmentation...  ')
        augmentation(X, y)

    print('shuffle images...')
    X, y = shuffle(X, y)
    X_test, y_test = shuffle(X_test, y_test)

    for shape in shapes:
        print('testing shape', shape, '...')
        acc, t = preprocess_and_train(X, y, X_test, y_test, shape=shape)
        print('accuracy = {}; time = {}. ({})'.format(acc, t, title))
        accuracies.append(acc)
        times.append(t)

    plt.plot(shapes, accuracies)
    plt.ylabel('accuracy')
    plt.xlabel('shape')
    plt.title(title)
    plt.show()
    plt.plot(shapes, times)
    plt.ylabel('time')
    plt.xlabel('shape')
    plt.title(title)
    plt.show()


def main():
    print('reading images...  ', end='')
    imgs, lbls = read_traffic_signs('training/Images')
    print('({})'.format(len(imgs)))

    print('splitting onto train/val sets...  ', end='')
    X, y, X_test, y_test = split_and_frequency(imgs, lbls)
    print(len(X), 'in train set.')

    experiment_and_analyze(X, y, X_test, y_test, augment=False, title='without augmentation')

    experiment_and_analyze(X, y, X_test, y_test, augment=True, title='with augmentation')


if __name__ == '__main__':

    main()
