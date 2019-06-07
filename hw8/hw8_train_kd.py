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
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from keras.utils import np_utils

criterion = 'categorical_crossentropy'
opt = Adam(lr = 1e-3)

def readTrainData(path):
    print("Reading File...")
    x_train = []

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(48, 48, 1)
        x_train.append(tmp)

    x_train = np.array(x_train, dtype=float) / 255.0

    return x_train
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
    model.summary()
    checkpoint = ModelCheckpoint(os.path.join(path, 'best.h5'), monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True, mode = 'max')
    
    return model, checkpoint

class kd_generator():
    def __init__(self, generator, batch_size):
        self.generator = generator
        self.batch_size = batch_size
        self.models = []
        self.models.append(load_model('best_ensemble1.h5'))
        self.models.append(load_model('best_0.69155ensemble2.h5'))
        self.models.append(load_model('best_0.68013ensemble3.h5'))
        self.models.append(load_model('best_0.69072ensemble4.h5'))
        self.models.append(load_model('best_ensemble5.h5'))
    def predict(self, data):
        ans = []
        for model in self.models:
            ans.append(model.predict(data))
        predict = np.array(ans[0] + ans[1] + ans[2] + ans[3] + ans[4]) / len(ans)
        return predict
    def __next__(self):
        data = next(self.generator)
        label = self.predict(data)
        return data, label


def train(model, checkpoint, x_train, epoch, path):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=0.,  # epsilon for ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.,
        shear_range=0.2,  # set range for random shear
        zoom_range=0.2,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    datagen.fit(x_train)
    sample_per_epoch = x_train.shape[0] * 8
    kd_gen = kd_generator(datagen.flow(x_train, batch_size=256), 256)
    model.fit_generator(kd_gen,
                        epochs=epoch,
                        samples_per_epoch=sample_per_epoch,
                        callbacks = [checkpoint],
                        workers = 0)
    model.save_weights(os.path.join(path, 'weight.h5'))

def main(arg):
    x_train = readTrainData(arg.train_data)
    model, checkpoint= build_model(args.out_dir)
    model.compile(loss = criterion, optimizer = opt, metrics = ['accuracy'])
    train(model, checkpoint, x_train, args.epoch, args.out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW8 training')
    parser.add_argument('train_data', type = str)
    parser.add_argument('out_dir', type = str)
    parser.add_argument('--epoch', default = 50, type = int)
    args = parser.parse_args()
    main(args)
