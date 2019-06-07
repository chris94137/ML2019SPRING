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


def build_model():
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
    model.summary()
    return model
def weight(model):
    weights = []
    for layer in model.layers:
        weights.append([w.astype(np.float16) for w in layer.get_weights()])
    return weights
def main(args):
    model = build_model()
    model.load_weights(args.model)
    weights = weight(model)
    np.savez_compressed(os.path.join(args.out_dir, 'weight.npz'), weights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW8 compressing')
    parser.add_argument('model', type = str)
    parser.add_argument('out_dir', type = str)
    args = parser.parse_args()
    main(args)