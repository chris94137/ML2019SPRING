import csv
import math

import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import Dense  # Fully Connected Networks
from keras.layers import MaxPooling2D  # Pooling
from keras.layers import (Activation, BatchNormalization, Dropout, Flatten,
                          Input)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, load_model
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from keras.utils import np_utils
import sys

delete_list = [59, 223, 418, 2059, 2171, 2809, 2892, 3262, 3931, 4275, \
               5274, 5439, 5509, 5722, 5881, 6102, 6458, 6893, 7172, 7496, \
               7527, 7629, 8030, 8737, 8856, 9026, 9500, 9679, 10585, 11244, \
               11286, 11295, 11846, 12289, 12352, 13148, 13988, 14279, 15144, 15838, \
               15894, 16540, 17081, 18012, 19238, 19632, 20222, 20712, 20817, 21817, \
               22198, 22927, 23596, 23894, 24053, 24408, 24891, 25219, 25603, 25909, \
               26383, 26561, 26860, 26897, 27292]
not_obvious_list = [115, 164, 361, 394, 427, 761, 2658, 3068, 4132, 4421, \
                    4860, 5750, 5789, 6309, 6424, 6426, 6478, 6624, 6970, 8423, \
                    12284, 12730, 15389, 15835, 17879, 18749, 21293, 22420, 22671, 23690,\
                    24191, 24333, 28153]
655
def readTrainData(file_name_X = '/content/train.csv'):
    data = pd.read_csv(file_name_X, encoding = 'Big5').to_numpy()
    x_data = data[:, 1:]
    y_data = data[:, 0]
    x_int_data = []
    for i in range(x_data.shape[0]):
        x_int_data.append(str.split(x_data[i][0]))
    x_int_data = np.array(x_int_data)
    y_data = np.array(y_data)
    x_int_data = x_int_data.astype('float')
    y_data = y_data.astype('float')
    x_train = []
    for i in range(x_int_data.shape[0]):
        x_train.append(x_int_data[i].reshape((48, 48, 1)))
    x_train = np.array(x_train)
    x_train = x_train / 255
    return_x = []
    return_y = []
    for i in range(len(x_train)):
        # if i not in delete_list:
        if True:
            return_x.append(x_train[i])
            return_y.append(y_data[i])
    return_y = np_utils.to_categorical(return_y)
    return_x = np.array(return_x)
    return_y = np.array(return_y)
    return return_x, return_y
def build_model():
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), input_shape = (48, 48, 1), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (5, 5), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(filters = 128, kernel_size = (3, 3), strides = 1, activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), strides = 1, activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(units = 512, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # model.add(Dense(units = 512, activation = 'relu', kernel_regularizer = l2(0.01)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(BatchNormalization())
    # model.add(BatchNormalization())
    # act = LeakyReLU(alpha=0.3)
    # model.add(Dense(units = 512, kernel_regularizer = l2(0.01)))
    # model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(units = 7, activation = 'softmax'))
    model.summary()
    checkpoint = ModelCheckpoint('best.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
    return model, checkpoint
if __name__ == '__main__':
    x_train, y_train = readTrainData(sys.argv[1])
    model, checkpoint= build_model()
    # opt = rmsprop(lr = 0.0001, decay = 1e-6)
    opt = Adam(lr = 5e-4)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    # model.fit(x_train, y_train, batch_size = 32, epochs = 200, validation_split = 0.15, callbacks = [checkpoint])
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # zca_epsilon=1e-06,  # epsilon for ZCA whitening
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
    val_rate = 0.1
    x_val = x_train[int(len(x_train) * (1 - val_rate)) : ]
    x_train = x_train[ : int(len(x_train) * (1 - val_rate))]
    y_val = y_train[int(len(y_train) * (1 - val_rate)) : ]
    y_train = y_train[ : int(len(y_train) * (1 - val_rate))]
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    sample_per_epoch = len(y_train) * 8
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=64),
                        epochs=100,
                        validation_data=(x_val, y_val),
                        samples_per_epoch=sample_per_epoch,
                        callbacks = [checkpoint])
    model.save('keras_model.h5')
