import csv
import math

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import Dense  # Fully Connected Networks
from keras.layers import MaxPooling2D  # Pooling
from keras.layers import (Activation, BatchNormalization, Dropout, Flatten,
                          Input)
from keras.models import Sequential, load_model
from keras.optimizers import *
import sys

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
def save_predict(ans, file_name = 'ans.csv'):
    file = open(file_name, 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'label'])
    for i in range(len(ans)):
        writer.writerow([str(i), ans[i]])
    file.close()

if __name__ == '__main__':
    x_test = readTestData(sys.argv[1])
    model = load_model('best_ensemble1.h5')
    ans1 = []
    for arr in x_test:
        ans1.append(np.squeeze(model.predict(np.expand_dims(arr, 0))).tolist())

    model = load_model('best_0.69155ensemble2.h5')
    ans2 = []
    for arr in x_test:
        ans2.append(np.squeeze(model.predict(np.expand_dims(arr, 0))).tolist())

    model = load_model('best_0.68013ensemble3.h5')
    ans3 = []
    for arr in x_test:
        ans3.append(np.squeeze(model.predict(np.expand_dims(arr, 0))).tolist())

    model = load_model('best_0.69072ensemble4.h5')
    ans4 = []
    for arr in x_test:
        ans4.append(np.squeeze(model.predict(np.expand_dims(arr, 0))).tolist())

    model = load_model('best_ensemble5.h5')
    ans4 = []
    for arr in x_test:
        ans4.append(np.squeeze(model.predict(np.expand_dims(arr, 0))).tolist())
        
    predict = []
    for index in range(len(ans1)):
        ls = (np.array(ans1[index]) + np.array(ans2[index]) + np.array(ans3[index]) + np.array(ans4[index])).tolist()
        predict.append(ls.index(max(ls)))
    save_predict(predict, sys.argv[2])