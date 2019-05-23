import os
import sys
import csv

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import (GRU, LSTM, Activation, BatchNormalization, Dense,
                          Dropout, Embedding, Input, LeakyReLU, GaussianNoise)
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from skimage import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.preprocessing.image import ImageDataGenerator

def read_img(path):
    img_list = []
    for i in range(1, 40001):
        img = io.imread(os.path.join(path, '0' * (6 - len(str(i))) + str(i) + '.jpg'))
        img_list.append(img)
    img_list = np.array(img_list)
    img_list = img_list.astype('float') / 255
    return img_list
def build_model(x_train, input_dim):
    input_ = Input(shape = input_dim)

    encoded = Conv2D(32, (3, 3), padding='same')(input_)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)

    encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    encoded = Conv2D(16, (3, 3), padding='same')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)

    encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    # encoded = Conv2D(16, (3, 3), padding='same')(encoded)
    # encoded = BatchNormalization()(encoded)
    # encoded = LeakyReLU()(encoded)
    # encoded = MaxPooling2D((2, 2), padding='same')(encoded)


    encoded = Flatten()(encoded)

    encoded = Dense(1024)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(512)(encoded)
    encoded = BatchNormalization()(encoded)

    #========================================================================#

    decoded = Dense(512)(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(1024)(decoded)
    decoded = BatchNormalization()(decoded)

    decoded = Reshape((8, 8, -1))(decoded)
    # decoded = Conv2D(16, (3, 3), padding='same')(encoded)
    # decoded = BatchNormalization()(decoded)
    # decoded = LeakyReLU()(decoded)

    # decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(16, (3, 3), padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)
    
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(32, (3, 3), padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)

    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('sigmoid')(decoded)

    encoder = Model(input_, encoded)
    auto_encoder = Model(input_, decoded)

    opt = Adam()

    auto_encoder.compile(loss = 'binary_crossentropy', optimizer = opt)
    auto_encoder.summary()
    # auto_encoder.fit(x_train, x_train, batch_size = 512, epochs = 80, validation_split = 0.1)

    #================================= img data gen =================================#
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
    sample_per_epoch = x_train.shape[0] * 8
    x_val = x_train[int(len(x_train) * (1 - val_rate)) : ]
    x_train = x_train[ : int(len(x_train) * (1 - val_rate))]
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    print('sample per epoch :', sample_per_epoch)
    # Fit the model on the batches generated by datagen.flow().
    auto_encoder.fit_generator(datagen.flow(x_train, x_train,
                                     batch_size=256),
                        epochs=100,
                        validation_data=(x_val, x_val),
                        samples_per_epoch=sample_per_epoch)
    encoder.save('encoder.h5')
    auto_encoder.save('auto_encoder.h5')
    return encoder, auto_encoder
def show_img(model, path, file):
    img = io.imread(os.path.join(path, file))
    img = img.astype('float') / 255
    output_img = model.predict(np.expand_dims(img, 0))
    output_img = (output_img * 255).astype(np.uint8)
    img = (img * 255).astype(np.uint8)
    output_img = output_img.squeeze()
    io.imsave('re_' + file, output_img)
    io.imsave('or_' + file, img)
def cluster(img_list, encoder, file):
    encoded_img = encoder.predict(img_list)
    encoded_img = encoded_img.reshape(encoded_img.shape[0], -1)
    pca = PCA(n_components=512, whiten=True, random_state=87).fit_transform(encoded_img)
    k = KMeans(n_clusters=2, random_state=87).fit(pca)
    print('result:', k.labels_.sum() / len(k.labels_))
    test_file = csv.reader(open(file))
    compare = list(test_file)[1:]
    ans = []
    for pair in compare:
        ans.append(int(k.labels_[int(pair[1]) - 1] == k.labels_[int(pair[2]) - 1]))
    return ans
def save_predict(ans, file_name = 'ans.csv'):
    file = open(file_name, 'w')
    writer = csv.writer(file, delimiter = ',', lineterminator = '\n')
    writer.writerow(['id', 'label'])
    for i in range(len(ans)):
        writer.writerow([str(i), ans[i]])
    file.close()
if __name__ == "__main__":
    img_list = read_img(sys.argv[1])
    # encoder, auto_encoder = build_model(img_list, (img_list.shape[1], img_list.shape[2], img_list.shape[3]))
    encoder = load_model('encoder_best.h5')
    # auto_encoder = load_model('auto_encoder_best.h5')
    # show_img(auto_encoder, sys.argv[1], '000002.jpg')
    ans = cluster(img_list, encoder, sys.argv[2])
    save_predict(ans, sys.argv[3])
    # print(len(ans))

    

    

