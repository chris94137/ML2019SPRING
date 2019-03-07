import numpy as np
import pandas as pd
import csv

def readTrainData(file_name = 'train.csv'):
    file_data = pd.read_csv(file_name, encoding = 'Big5').to_numpy()
    data = file_data[ : , 3 : ]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 'NR':
                data[i][j] = 0
    data = data.astype('float')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] < 0:
                if j != data.shape[1] - 1:
                    if j != 0:
                        data[i][j] = (data[i][j - 1] + data[i][j + 1]) / 2
                    else:
                        data[i][j] = data[i][j + 1]
                else:
                    if j != 0:
                        data[i][j] = data[i][j - 1]
    X, Y = [], []
    for i in range(0, data.shape[0], 18 * 20): # operate data of one month
        month_data = np.vsplit(data[i : i + 18 * 20], 20)
        concat = np.concatenate(month_data, axis = 1)
        for j in range(concat.shape[1] - 9):
            features = concat[ : , j : j + 9].reshape(-1)
            features = np.append(features, [1])
            X.append(features)
            Y.append(concat[9][j + 9])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
def readTestData(file_name = 'test.csv'):
    file_data = pd.read_csv(file_name, encoding = 'Big5', header = None).to_numpy()
    data = file_data[ : , 2 : ]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] == 'NR':
                data[i][j] = 0
    data = data.astype('float')
    data = np.vsplit(data, data.shape[0] // 18)
    X = []
    for arr in data:
        features = arr.reshape(-1)
        features = np.append(features, [1])
        X.append(features)
    return X
def train(x, y):
    lr = 0.1
    iteration = 100000
    w = np.zeros(x.shape[1])
    grad_sum = np.zeros(x.shape[1])
    prev_cost = 2 ** 31 - 1
    lumbda = 1
    for i in range(iteration):
        y_prime = np.dot(x, w)
        loss = y_prime - y
        cost = np.sum(loss ** 2 / len(loss))
        grad = 2 * (np.dot(x.T, loss) + lumbda * w)
        grad_sum += grad ** 2
        ada = np.sqrt(grad_sum)
        w = w - lr * grad / ada
        print("\riteration :", format(i, ">10d"), "| cost :", cost, end = '')
        if abs(prev_cost - cost) < 0.00001:
            break
        else:
            prev_cost = cost
    print("\ndone")
    return w



if __name__ == '__main__':
    train_x, train_y = readTrainData()
    w = train(train_x, train_y)
    test_x = readTestData()
    ans = []
    for arr in test_x:
        ans.append(np.dot(arr, w))
    file = open('ans.csv', 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'value'])
    for i in range(len(ans)):
        writer.writerow(['id_' + str(i), ans[i]])
    file.close()
        

    

