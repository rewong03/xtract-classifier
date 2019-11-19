import os
import csv
import itertools
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt


# Functions for getting features from files/directories

def feature_from_file(file_path, feature_type="head", byte_num=512):  # will add more feature_type later
    """Retrieves features from a file.

    Parameters:
    feature_type (str): "head" to get bytes from head of the file.
    byte_num (int): Number of bytes to grab.
    file_path (str): File path of file to get features from.

    Returns:
    List of bytes from file_path.
    """
    if feature_type == "head":
        with open(file_path, 'rb') as f:
            byte = f.read(1)
            index = 1
            features = []

            while byte and index <= byte_num:
                features.append(byte)
                index += 1
                byte = f.read(1)

            if len(features) < byte_num:
                features.extend([b'' for i in range(byte_num - len(features))])

            assert len(features) == byte_num
            return features
    else:
        print("Invalid feature type")


def feature_from_dir(dir_path, feature_type="head", byte_num=512):# :( can't figure out how to implement multiprocessing with multiple parameters
    """Takes a directory and grabs features from each file.

    Parameters:
    dir_path (str): Path of directory to take features from.
    feature_type (str): Type of features to get.
    byte_num (str): Number of features to take

    Return:
    features (list): 2D list of byte_num bytes from each fie in dir_path.
    """
    file_paths = []
    features = []

    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))

    pools = mp.Pool()

    for feature in pools.imap(feature_from_file, file_paths):
        features.append(feature)

    pools.close()
    pools.join()

    return features


def translate_bytes(dir_features):
    """Translates bytes into integers.

    Parameter:
    dir_features (list): 2D list of bytes.

    Return:
    translated_features (numpy array): dir_features with bytes translated to integers.
    """
    translated_features = np.zeros((len(dir_features), len(dir_features[0])))

    for idx, file_features in enumerate(dir_features):
        x = np.array([int.from_bytes(c, byteorder="big") for c in file_features])
        translated_features[idx] = x

    return translated_features


def grab_labels(csv_path):
    """Returns the file paths and file labels from a naivetruth csv.

    Parameter:
    csv_path (str): Path of csv file to take labels and paths from.

    Returns:
    labels (list): List of label strings from csv_path.
    file_paths (list): List of file_paths from csv_path.
    """
    labels = []
    file_paths = []

    with open(csv_path) as label_file:
        csv_reader = csv.reader(label_file, delimiter=',')
        for row in csv_reader:
            file_paths.append(row[0])
            labels.append(row[2])

    return labels, file_paths


def features_from_list(file_paths):
    """Grabs features from a list of file paths.

    Parameter:
    file_paths (str): List of file paths to get features from.

    Return:
    features (list): List of features from files.
    """
    features = []

    pools = mp.Pool()

    for feature in pools.imap(feature_from_file, file_paths):
        features.append(feature)

    pools.close()
    pools.join()

    return features


# Functions to create a confusion matrix


def convert_to_index(array_categorical):
    """Turns a list of numpy array ouputted from a categorical classifier into a
    single integer.

    Parameter:
    array_categorical (list): List of numpy arrays ouputted from a categorical classifier.

    Return:
    array_index (list): List of integers.
    """
    array_index = [np.argmax(array_temp) for array_temp in array_categorical]
    return array_index


def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function modified to plots the ConfusionMatrix object.
    Normalization can be applied by setting `normalize=True`.

    Code Reference :
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This script is derived from PyCM repository: https://github.com/sepandhaghighi/pycm

    """

    plt_cm = []
    for i in cm.classes:
        row = []
        for j in cm.classes:
            row.append(cm.table[i][j])
        plt_cm.append(row)
    plt_cm = np.array(plt_cm)
    if normalize:
        plt_cm = plt_cm.astype('float') / plt_cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(plt_cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm.classes))
    plt.xticks(tick_marks, cm.classes, rotation=45)
    plt.yticks(tick_marks, cm.classes)

    fmt = '.2f' if normalize else 'd'
    thresh = plt_cm.max() / 2.
    for i, j in itertools.product(range(plt_cm.shape[0]), range(plt_cm.shape[1])):
        plt.text(j, i, format(plt_cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if plt_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predict')
