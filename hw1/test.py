import numpy as np
import pandas as pd
import csv
import sys

def readTestData(file_name = 'test.csv'):
    file_data = pd.read_csv(file_name, encoding = 'Big5', header = None).to_numpy()
    data = file_data[ : , 2 : ]
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
    data = np.vsplit(data, data.shape[0] // 18)
    X = []
    for arr in data:
        features = arr.reshape(-1)
        features = np.append(features, [1])
        X.append(features)
    return X
if __name__ == '__main__':
    w = np.load('model.npy')
    test_x = readTestData(sys.argv[1])
    ans = []
    for arr in test_x:
        ans.append(np.dot(arr, w))
    file = open(sys.argv[2], 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'value'])
    for i in range(len(ans)):
        writer.writerow(['id_' + str(i), ans[i]])
    file.close()
        