import numpy as np
from skimage import io, data
import os, sys

def read_data(path):
    img_list = []
    for i in range(415):
        img = io.imread(os.path.join(path, str(i) + '.jpg'))
        img_list.append(img.flatten())
    img_list = np.array(img_list)
    mean = np.mean(img_list, axis = 0)  
    x = img_list - mean
    return x, mean
def get_SVD(x):
    u, sigma, v = np.linalg.svd(x.T, full_matrices = False)
    return u, sigma, v
def save_eigenface(u):
    for i in range(10):
        M = np.reshape(u.T[i], (600, 600, 3))
        M -= np.min(M)
        M /= np.max(M)
        M = (M * 255).astype(np.uint8)
        io.imsave("./eigenface/eg_" + str(i) + ".jpg", M)
def reconstruct(file, U, mean, name):
    img = io.imread(file)
    img = img.flatten()
    img = img.astype('float')
    img -= mean
    u = U[:, :5]
    w = u.T.dot(img.T)
    res = u.dot(w)
    M = np.reshape(res.T, (600, 600, 3))
    M += mean.reshape((600, 600, 3))
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    io.imsave(name, M)
def ratio(s):
    for i in range(5):
        number = s[i] * 100 / sum(s)
        print(number)
def mean_face(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    io.imsave('average.jpg', M.reshape((600, 600, 3)))  
if __name__ == "__main__":
    x, mean = read_data(sys.argv[1])
    u, sigma, v = get_SVD(x)
    reconstruct_file = os.path.join(sys.argv[1], sys.argv[2])
    reconstruct(reconstruct_file, u, mean, sys.argv[3])
#     save_eigenface(u)
#     ratio(sigma)
#     mean_face(mean)
    
    