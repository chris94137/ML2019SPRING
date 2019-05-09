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
from keras.layers import (LSTM, Activation,  # Fully Connected Networks
                          BatchNormalization, Dense, Dropout)
from keras.models import Sequential, load_model
from keras.optimizers import Adam, Nadam
from keras.preprocessing.sequence import pad_sequences

constraint_len = 30

def read_test_data(file_name_test = './test_x.csv'):
    dm = pd.read_csv(file_name_test)
    data = dm['comment']
    P = Pool(processes=4) 
    data = P.map(tokenize, data)
    P.close()
    P.join()
    test_x = data

    return test_x
def tokenize(sentence):
    tokens = jieba.lcut(sentence)
    return tokens
def load_embed(file_name = './embed.model'):
    print('start loading word model')
    model = word2vec.Word2Vec.load(file_name)
    return model
def save_predict(ans, file_name):
    print('start saving...')
    file = open(file_name, 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'label'])
    for i in range(len(ans)):
        writer.writerow([str(i), ans[i]])
    file.close()
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
def test(test_x, word_model, model):
    print('start testing...')
    word_list = {}
    vocab_list = [word for word, vector in word_model.wv.vocab.items()]
    for i, word in enumerate(vocab_list):
        word_list[word] = i + 1   
    
    test_x = text_to_index(test_x, word_list)
    predict = model.predict(test_x)
    ans = []
    for i in range(len(predict)):
        ans.append(1 if predict[i] >= 0.5 else 0)
    print('done')
    return ans
def main(args):
    jieba.load_userdict(args.jieba_lib)
    test_x = read_test_data(args.test_X)
    model = load_model(args.model)
    word_model = load_embed()
    ans = test(test_x, word_model, model)
    save_predict(ans, args.ans_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ans_dir', type=str, help='[Output] Your ans directory')
    parser.add_argument('model', type=str, help='[Output] Your model')
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('test_X',type=str, help='[Input] Your test_x.csv')
    parser.add_argument('word_model', type=str)

    args = parser.parse_args()
    main(args)
