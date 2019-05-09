import argparse
import csv
import os
import sys
from multiprocessing import Pool

import jieba
import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras.callbacks import ModelCheckpoint
from keras.layers import (GRU, LSTM, Activation, BatchNormalization, Dense,
                          Dropout, Embedding)
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Nadam
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

opt = Adam()

def read_train_data(file_name_x = './train_x.csv', file_name_y = './train_y.csv', file_name_test = './test_x.csv'):
    dm = pd.read_csv(file_name_x)
    data = dm['comment']
    data = data[:119018]
    P = Pool(processes=4) 
    data = P.map(tokenize, data)
    P.close()
    P.join()
    train_x = data

    dm = pd.read_csv(file_name_y)
    train_y = [int(i) for i in dm['label']]
    train_y = train_y[:119018]

    dm = pd.read_csv(file_name_test)
    data = dm['comment']
    P = Pool(processes=4) 
    data = P.map(tokenize, data)
    P.close()
    P.join()
    test_x = data

    return train_x, train_y, test_x

def tokenize(sentence):
    tokens = jieba.lcut(sentence)
    return tokens

def embed(train_x, test_x, file_path, jieba_lib):
    print('start training word model, using library', jieba_lib)
    model = word2vec.Word2Vec(train_x + test_x, size=250, iter=10, sg=1, window=5, min_count=5, workers=8)
    print('finish...')
    model.save(os.path.join(file_path, 'embed.model'))
    return model
def load_embed(file_name = './embed.model'):
    print('start loading word model')
    model = word2vec.Word2Vec.load(file_name)
    return model
def text_to_index(data, word_list):
    index_data = []
    for sentence in data:
        new_sentence = []
        for word in sentence:
            try:
                new_sentence.append(word_list[word])
            except:
                new_sentence.append(0)
        index_data.append(new_sentence)
    index_data = pad_sequences(index_data, maxlen = 200)
    return np.array(index_data)
def train(train_x,train_y,word_model, model_path):
    embedding_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size))
    word_list = {}
    vocab_list = [(word, word_model.wv[word]) for word, vector in word_model.wv.vocab.items()]
    for i, vocab in enumerate(vocab_list):
        word, vec = vocab
        embedding_matrix[i + 1] = vec
        word_list[word] = i + 1

    train_x = text_to_index(train_x, word_list)

    embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                output_dim=embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                trainable=True)
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(32, return_sequences = True))
    model.add(GRU(32))
    # model.add(Dropout(0.25))
    # model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.summary()

    model.compile(optimizer= opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    checkpoint1 = ModelCheckpoint(os.path.join(model_path, 'best_loss.h5'), monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
    checkpoint2 = ModelCheckpoint(os.path.join(model_path, 'best_acc.h5'), monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
    callbacks_list = [checkpoint1, checkpoint2]

    epochs = 20
    batch_size = 512

    historys = model.fit(train_x, train_y, 
                        batch_size=batch_size,
                        epochs=epochs, 
                        validation_split=0.1,
                        callbacks=callbacks_list)

    history_save = pd.DataFrame(data=historys.history)
    history_save.to_csv(os.path.join(model_path, 'rnn_history.csv'),index=False)
    model.save(os.path.join(model_path, 'model.h5'))

def main(args):
    jieba.load_userdict(args.jieba_lib)
    train_x, train_y, test_x = read_train_data(args.train_X, args.train_Y, args.test_X)
    # word_model = embed(train_x,test_x, args.model_dir, args.jieba_lib)
    word_model = load_embed()
    train(train_x, train_y, word_model, args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str, help='[Output] Your model checkpoint directory')
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('train_X',type=str, help='[Input] Your train_x.csv')
    parser.add_argument('train_Y',type=str, help='[Input] Your train_y.csv')
    parser.add_argument('test_X',type=str, help='[Input] Your test_x.csv')

    args = parser.parse_args()
    main(args)
