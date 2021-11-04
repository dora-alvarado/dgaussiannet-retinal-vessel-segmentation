import cv2
import os
import numpy as np
from PIL import Image
from .quantum_layer import QuantumLayer
import pickle

def change_range(data, input_min, input_max, output_min, output_max, eps=1e-8):
    result = ((data - input_min) / (input_max - input_min+eps)) * (output_max - output_min) + output_min
    return result


def read_grayscale_img(path):
    img = Image.open(path).convert('L')
    m, n = img.getdata().size
    img = np.asarray(img.getdata()).reshape(n, m, 1)
    return img

def read_preproc_img(path, h, w):
    img = Image.open(path).convert('L')
    #m, n = img.getdata().size
    img = np.asarray(img.getdata()).reshape(h, w, -1)
    return img

def clahe_equalized(img):
    assert (len(img.shape) == 2)  # 2D image
    assert issubclass(img.dtype.type, np.uint8) # unsigned int 8 bits
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #create a CLAHE object
    img_equalized = clahe.apply(img)
    return img_equalized


def adjust_gamma(img, gamma=1.0):
    assert issubclass(img.dtype.type, np.uint8)  # unsigned int 8 bits
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_img = cv2.LUT(img, table)
    return new_img


def qrandc(img, kernel_size=3, stride=1, qparams =None, qparams_path='./qparams.pkl'):
    assert (len(img.shape) == 3 or len(img.shape)==2)  # 2D arrays (H, W) or 3D arrays (H, W, C)
    if len(img.shape) == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1
    z = QuantumLayer(depth=c, kernel_size=kernel_size, stride=stride)
    if qparams is not None:
        z.rand_params = qparams
    dir_path = os.path.dirname(os.path.abspath(qparams_path))
    basename = os.path.basename(qparams_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(qparams_path, 'wb') as file:
        pickle.dump(z.rand_params, file)
    dir_path_test = dir_path.replace('training', 'test')
    if not os.path.exists(dir_path_test):
        os.makedirs(dir_path_test)
    print(dir_path_test+'/'+basename)
    with open(dir_path_test+'/'+basename, 'wb') as file:
        pickle.dump(z.rand_params, file)
    new_img_ = z.quanv(img)
    new_img_ = new_img_.reshape((h, w, kernel_size ** 2))
    new_img_ = change_range(new_img_, new_img_.min(), new_img_.max(), 0., 1.)
    return new_img_


def save_img(img, path):
    dir_path = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    cv2.imwrite(path, img)
