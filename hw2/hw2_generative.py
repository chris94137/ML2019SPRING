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
    # sigma = np.std(X_train, axis = 0)
    # for i in range(X_train.shape[1]):
    #     X_train[:, i] = X_train[:, i] / sigma[i]
    X1 = []
    X2 = []
    for i in range(len(Y_train)):
        if Y_train[i] == 1:
            X1.append(X_train[i])
        else:
            X2.append(X_train[i])
    X1 = np.array(X1)
    X2 = np.array(X2)
    return X1, X2, Y_train
def readTestData(file_name = 'X_test'):
    X_test = pd.read_csv(file_name, encoding = 'Big5').to_numpy()
    X_test = X_test.astype('float')
    X_test = np.array(X_test)
    # sigma = np.std(X_test, axis = 0)
    # for i in range(X_test.shape[1]):
    #     if sigma[i] != 0:
    #         X_test[:, i] = X_test[:, i] / sigma[i]
    #     else:
    #         X_test[:, i] = 0
    return X_test
def sigmoid(y):
    return 1 / (1 + np.exp(-y))
def gaussian(x, u, sigma):
    a = 1 / ((2 * math.pi) ** (len(x) / 2))
    b = 1 / (np.linalg.det(sigma) ** 0.5)
    c = np.exp(- 0.5 * np.dot(np.dot((x - u.T), np.linalg.inv(sigma)), (x - u.T).T))
    return a * b * c
def compute_matrix(x1, x2, y, file_name = 'model.npy'):
    u_x1 = np.sum(x1, axis = 0) / x1.shape[0]
    error1 = x1 - u_x1.T
    sigma1 = np.dot(error1.T, error1) / x1.shape[0]
    u_x2 = np.sum(x2, axis = 0) / x2.shape[0]
    error2 = x2 - u_x2.T
    sigma2 = np.dot(error2.T, error2) / x2.shape[0]
    sigma = (x1.shape[0] / len(y)) * sigma1 + (x2.shape[0] / len(y)) * sigma2
    return u_x1, u_x2, sigma, x1.shape[0], x2.shape[0]
def predict(x, u1, u2, sigma, N1, N2):
    ans = []
    p1 = N1 / (N1 + N2)
    p2 = N2 / (N1 + N2)
    for data in x:
        f1 = gaussian(data, u1, sigma)
        f2 = gaussian(data, u2, sigma)
        if f1 == 0:
            prob = 0
        elif f2 == 0:
            prob = 1
        else:
            prob = (f1 * p1) / (f1 * p1 + f2 * p2)
        if prob > 0.5:
            ans.append(1)
        else:
            ans.append(0)
    return ans
def save_predict(ans, file_name = 'ans.csv'):
    file = open(file_name, 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'label'])
    for i in range(len(ans)):
        writer.writerow([str(i + 1), ans[i]])
    file.close()
if __name__ == '__main__':
    x_train1, x_train2, y_train = readTrainData(sys.argv[1], sys.argv[2])
    u1, u2, sigma, N1, N2 = compute_matrix(x_train1, x_train2, y_train)
    x_test = readTestData(sys.argv[3])
    ans = predict(x_test, u1, u2, sigma, N1, N2)
    save_predict(ans, sys.argv[4])
