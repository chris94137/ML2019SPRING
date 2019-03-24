import csv
import math
import sys

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input
from keras.models import Sequential, load_model
from keras.optimizers import *

def readTestData(file_name = 'X_test'):
    X_test = pd.read_csv(file_name, encoding = 'Big5').to_numpy()
    X_test = X_test.astype('float')
    X_test = np.array(X_test)
    sigma = np.std(X_test, axis = 0)
    for i in range(X_test.shape[1]):
        mean = np.sum(X_test[:, i]) / X_test.shape[0]
        if sigma[i] != 0:
            X_test[:, i] = (X_test[:, i] - mean) / sigma[i]
        else:
            X_test[:, i] = 0
    bias = np.ones(X_test.shape[0])
    X_test = np.column_stack((X_test, bias))
    X_test = np.delete(X_test, [31, 37], axis = 1) #marital status
    # X_test = np.delete(X_test, [71, 107], axis = 1) #country
    return X_test
def save_predict(ans, file_name = 'ans.csv'):
    file = open(file_name, 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'label'])
    for i in range(len(ans)):
        writer.writerow([str(i + 1), ans[i]])
    file.close()

if __name__ == '__main__':
    model = load_model('best(0.86093).h5')
    x_test = readTestData(sys.argv[1])
    ans = []
    for arr in x_test:
        ans.append(model.predict(np.expand_dims(arr, 0)))
    predict = []
    for i in ans:
        if i >= 0.5:
            predict.append(1)
        else:
            predict.append(0)
    save_predict(predict, sys.argv[2])
