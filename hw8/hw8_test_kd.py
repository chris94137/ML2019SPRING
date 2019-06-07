import csv
import math
import sys
import argparse
import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, Input, MaxPooling2D, SeparableConv2D, AveragePooling2D, Lambda)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from keras.utils import np_utils


def readTestData(file_name = 'test.csv'):
    data = pd.read_csv(file_name, encoding = 'Big5').to_numpy()
    x_data = data[:, 1:]
    x_int_data = []
    for i in range(x_data.shape[0]):
        x_int_data.append(str.split(x_data[i][0]))
    x_int_data = np.array(x_int_data)
    x_int_data = x_int_data.astype('int')
    x_test = []
    for i in range(x_int_data.shape[0]):
        x_test.append(x_int_data[i].reshape((48, 48, 1)))
    x_test = np.array(x_test)
    x_test = x_test / 255
    return x_test
def build_model(path):
    model = Sequential()
    model.add(Conv2D(input_shape = (48, 48, 1), filters = 32, kernel_size = (3, 3), strides = (1, 1), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters = 64, kernel_size = (3, 3), strides = (2, 2), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(units = 84, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(units = 84, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Lambda(lambda x: x * 0.6))
    model.add(Dense(units = 7, activation = 'softmax'))
    return model
def save_predict(predict, out_file):
    file = open(out_file, 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'label'])
    for i, ans in enumerate(predict):
        writer.writerow([str(i), ans])
    file.close()
def load(model, npz):
    weights = np.load(npz)
    for i in range(len(weights['arr_0'])):
        model.layers[i].set_weights(weights['arr_0'][i])
    return model
def main(args):
    model= build_model(args.out_dir)
    model = load(model, args.model)
    x_test = readTestData(args.test_data)
    ans = model.predict(x_test)
    predict = np.argmax(ans, axis=1)
    save_predict(predict, args.out_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW8 testing')
    parser.add_argument('test_data', type = str)
    parser.add_argument('model', type = str)
    parser.add_argument('out_dir', type = str)
    args = parser.parse_args()
    main(args)