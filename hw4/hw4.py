import os
import keras
import pandas as pd
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf
import sys

out_path = sys.argv[2]
delete_list = [59, 2059, 2171, 2809]

def readTrainData(file_name_X = 'train.csv'):
    data = pd.read_csv(file_name_X, encoding = 'Big5').to_numpy()
    x_data = data[:, 1:]
    y_data = data[:, 0]
    x_int_data = []
    for i in range(x_data.shape[0]):
        x_int_data.append(str.split(x_data[i][0]))
    x_int_data = np.array(x_int_data)
    y_data = np.array(y_data)
    x_int_data = x_int_data.astype('float')
    y_data = y_data.astype('float')
    x_train = []
    for i in range(x_int_data.shape[0]):
        x_train.append(x_int_data[i].reshape((48, 48, 1)))
    x_train = np.array(x_train)
    x_train = x_train / 255
    return_x = []
    return_y = []
    for i in range(len(x_train)):
        # if i not in delete_list:
        if True:
            return_x.append(x_train[i])
            return_y.append(y_data[i])
    return_y = np_utils.to_categorical(return_y)
    return_x = np.array(return_x)
    return_y = np.array(return_y)
    return return_x, return_y
def plot_jpg(file_name, file, color_bar = False, title = False, title_str = ''):
    plt.imshow(g)
    if color_bar:
        cb = plt.colorbar()
    if title:
        plt.title(title_str)
    plt.savefig(file_name)
    if color_bar:
        cb.remove()
if __name__ == '__main__':
    path_list = [22, 416, 500, 7, 310, 494, 11]
    model = load_model("best_0.69155ensemble2.h5")
    x_train, y_train = readTrainData(sys.argv[1])
    images = [x_train[i] for i in path_list]
    y = [y_train[i] for i in path_list]
    images = np.array(images)
    y = np.array(y)
    sess = K.get_session()
    # Q1
    label = tf.placeholder(shape=(None, 7), dtype=tf.float64)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=model.output)
    cal_grad = tf.gradients(loss, model.input)
    grad = sess.run(cal_grad, feed_dict = {model.input:images, label:y})[0]
    for i in range(7):
        g = grad[i].reshape((48, 48))
        plot_jpg(out_path + 'fig1_' + str(i) + '.jpg', g, color_bar = True)
        # plt.imsave('fig1_' + str(i) + '.png', g)
        # mask = images[i]
        # max_ = np.max(grad[i])
        # mask[grad[i] < 0.05 * max_] = 0
        # mask = mask.reshape(48, 48)
        # plt.imsave('fig1_' + str(i) + '_mask.png', mask)
    
    # Q2
    images = [x_train[i] for i in path_list]
    y = [y_train[i] for i in path_list]
    images = np.array(images)
    y = np.array(y)
    # conv layer
    for index in range(model.layers[0].output.shape[1]):
        plt.subplot(8,8,index + 1)
        goal = model.layers[0].output[:, :, :, index]
        img = sess.run(goal, feed_dict={model.input:images[0].reshape(-1, 48, 48, 1)})[0]
        img = img.reshape(44, 44)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(out_path + 'fig2_2.jpg')

    # maxpooling layer
    for index in range(model.layers[4].output.shape[3]):
        plt.subplot(8,8,index + 1)
        goal = tf.reduce_mean(model.layers[4].output[:, :, :, index])
        x = np.random.randn(1,48,48,1)
        cal_grad = K.gradients(goal, model.input)

        epoch = 200
        lr = 10
        lr_x = 1e-7

        for i in range(epoch):
            grad, now_goal = sess.run([cal_grad, goal], feed_dict={model.input:x})
            grad = grad[0]
            lr_x += grad ** 2
            x += lr * grad / np.sqrt(lr_x)
        #     print('filter', index, now_goal, end = '\r', flush = True)
        # print()
        x = x.reshape(48, 48)
        plt.imshow(x)
        plt.axis('off')
    plt.savefig(out_path + 'fig2_1.jpg')

    # Q3
    images = [x_train[i] for i in path_list]
    y = [y_train[i] for i in path_list]
    images = np.array(images)
    y = np.array(y)
    explainer = lime_image.LimeImageExplainer()
    predict_ = lambda x : np.squeeze(model.predict(x[:, :, :, 0].reshape(-1, 48, 48, 1)))
    for i in range(7):
        image = [images[i]] * 3
        image = np.concatenate(image, axis = 2)
        np.random.seed(16)
        explanation = explainer.explain_instance(image, predict_, labels=(i, ), top_labels=None, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=True)
        plt.imsave(out_path + 'fig3_' + str(i) + '.jpg', mark_boundaries(temp / 2 + 0.5, mask))


    # Q4
    # bad_data = [x_train[i] for i in delete_list]
    # bad_data_y = [y_train[i] for i in delete_list]
    # bad_data = np.array(bad_data)
    # bad_data_y = np.array(bad_data_y)
    # label = tf.placeholder(shape=(None, 7), dtype=tf.float64)
    # loss_1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=model.output)
    # cal_grad_1 = tf.gradients(loss_1, model.input)
    # grad = sess.run(cal_grad_1, feed_dict = {model.input:bad_data, label:bad_data_y})[0]
    # for i in range(4):
    #     plt.subplot(4, 3, 3 * i + 1)
    #     plt.imshow(bad_data[i].reshape((48, 48)), cmap='gray')
    #     g = grad[i].reshape((48, 48))
    #     plt.subplot(4, 3, 3 * i + 2)
    #     plt.imshow(g, cmap='gray')
    #     mask = bad_data[i]
    #     max_ = np.max(grad[i])
    #     mask[grad[i] < 0.05 * max_] = 0
    #     mask = mask.reshape(48, 48)
    #     plt.subplot(4, 3, 3 * i + 3)
    #     plt.imshow(mask, cmap='gray')
    #     plt.axis('off')
    # plt.savefig('fig4_1.png')

    # bad_data = [x_train[i] for i in delete_list]
    # bad_data_y = [y_train[i] for i in delete_list]
    # bad_data = np.array(bad_data)
    # bad_data_y = np.array(bad_data_y)
    # model = load_model("best(0.68013ensemble3).h5")
    # loss_2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=model.output)
    # cal_grad_2 = tf.gradients(loss_2, model.input)
    # grad = sess.run(cal_grad_2, feed_dict = {model.input:bad_data, label:bad_data_y})[0]
    # for i in range(4):
    #     plt.subplot(4, 3, 3 * i + 1)
    #     plt.imshow(bad_data[i].reshape((48, 48)), cmap='gray')
    #     g = grad[i].reshape((48, 48))
    #     plt.subplot(4, 3, 3 * i + 2)
    #     plt.imshow(g, cmap='gray')
    #     mask = bad_data[i]
    #     max_ = np.max(grad[i])
    #     mask[grad[i] < 0.05 * max_] = 0
    #     mask = mask.reshape(48, 48)
    #     plt.subplot(4, 3, 3 * i + 3)
    #     plt.imshow(mask, cmap='gray')
    #     plt.axis('off')
    # plt.savefig('fig4_2.png')