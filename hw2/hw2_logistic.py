import numpy as np
import pandas as pd
import csv
import math
import sys

def readTrainData(file_name_X = 'X_train', file_name_Y = 'Y_train'):
    X_train = pd.read_csv(file_name_X, encoding = 'Big5').to_numpy()
    Y_train = pd.read_csv(file_name_Y, encoding = 'Big5').to_numpy()
    X_train = X_train.astype('float')
    Y_train = Y_train.astype('float')
    Y_train = np.concatenate(Y_train, axis = 0)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    sigma = np.std(X_train, axis = 0)
    for i in range(X_train.shape[1]):
        mean = np.sum(X_train[:, i]) / X_train.shape[0]
        X_train[:, i] = (X_train[:, i] - mean) / sigma[i]
    bias = np.ones(X_train.shape[0])
    X_train = np.column_stack((X_train, bias))
    return X_train, Y_train
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
    return X_test
def sigmoid(y):
    return 1 / (1 + np.exp(-y))
def train(x, y, file_name = 'model.npy'):
    lr = 0.05
    iteration = 20000
    w = np.zeros(x.shape[1])
    s_grad = np.zeros(x.shape[1])
    w += 0.01
    for i in range(iteration):
        y_prime = np.dot(x, w)
        f = sigmoid(y_prime)
        cross_entropy = -(y * np.log(f) + (1.0000000001 - y) * np.log(1.0000000001 - f))
        grad = np.dot(x.T, f - y)
        s_grad += grad ** 2
        ada = np.sqrt(s_grad)
        w -= lr * grad / ada
        predict = np.round(f)
        error = np.abs(predict - y)
        error = np.sum(error) / len(error)
        print("iteration :", format(i, ">10d"), "| entropy :", format(np.sum(cross_entropy) / len(cross_entropy), ">.15f"), "| accuracy :", 1 - error)
    np.save(file_name, w)
    return w
def predict(x, w):
    ans = []
    ans = np.dot(x, w)
    ans = sigmoid(ans)
    return ans
def save_predict(ans, file_name = 'ans.csv'):
    file = open(file_name, 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'label'])
    for i in range(len(ans)):
        writer.writerow([str(i + 1), ans[i]])
    file.close()
if __name__ == '__main__':
    # x_train, y_train = readTrainData()
    # w = train(x_train, y_train)
    w = np.load('model(0.85245).npy')
    x_test = readTestData(sys.argv[1])
    ans = predict(x_test, w)
    predict = []
    for i in ans:
        if i >= 0.5:
            predict.append(1)
        else:
            predict.append(0)
    save_predict(predict, sys.argv[2])
