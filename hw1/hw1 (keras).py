import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import  Adam
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib

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
def buildModel():
    model = Sequential()
    model.add(Dense(input_dim = 163, kernel_initializer = 'normal', units = 300, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(units = 30, activation = 'relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.1))
    # model.add(Dense(units = 8, activation = 'relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.15))
    model.add(Dense(units = 1, activation = 'linear'))
    model.summary()
    checkpoint = ModelCheckpoint('best.h5', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
    return model, checkpoint
    



if __name__ == '__main__':
    train_x, train_y = readTrainData()
    print(train_x.shape)
    print(train_y.shape)
    test_x = readTestData()
    model, checkpoint = buildModel()
    model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(train_x, train_y, batch_size = 50, epochs = 2000, validation_split = 0.15, callbacks = [checkpoint])
    model.save('keras_model.h5')
    model = load_model('best.h5')
    ans = []
    for arr in test_x:
        ans.append(model.predict(np.expand_dims(arr, 0)))
    file = open('keras_ans.csv', 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'value'])
    for i in range(len(ans)):
        writer.writerow(['id_' + str(i), ans[i][0][0]])
    file.close()
        

    

