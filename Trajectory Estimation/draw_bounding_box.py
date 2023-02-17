from glob import glob
from cv2 import threshold
from matplotlib import pyplot as plt
import cv2 as cv
import random
import numpy as np
from numpy.linalg import inv
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from time import time
from scipy.spatial.transform import Rotation as Rot
import math
from torch import scalar_tensor
import time
import copy
import gc

# from google.colab import drive, files
import os
import sys
import torch
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import sys
import os
import math
import numpy as np
import random
sys.path.append(os.path.abspath(''))
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LinearRegression
import datetime
import os
import matplotlib
from PIL import Image


from torch import tensor

def get_bounding_box(single_img:tensor) -> tuple:
    # single_img = single_img.numpy()
    if single_img.shape[0] == 3:
        single_img = single_img.permute(1,2,0)
    target = np.where(((single_img[:,:,0] < 255) & (single_img[:,:,1] < 255) & (single_img[:,:,2] < 255)))
    min_x = np.min(target[1])
    min_y = np.min(target[0])
    max_x = np.max(target[1])
    max_y = np.max(target[0])

    return ((min_x, min_y), (max_x, max_y))

# WARNING PIXEL VALUES ASSUME ORIGIN IS AT THE CENTER OF THE SCREEN 
def calculate_phi(p1, p2, p3, p1_new, p2_new, p3_new):
    
    affine_matrix = np.matmul(np.array(
        [
            [p1_new[0], p2_new[0], p3_new[0]],
            [p1_new[1], p2_new[1], p3_new[1]],
            [1.0, 1.0, 1.0] 
        ]
        ), np.linalg.inv(np.array([
        [p1[0], p2[0], p3[0]],
        [p1[1], p2[1], p3[1]],
        [1.0, 1.0, 1.0] 
        ])))
    phi = 1.0 / affine_matrix[0][0]
    return phi

if __name__ == '__main__':
    # data = torch.load("/home/tau/Video_Datasets/20Videos20Frames/2")[0]
    # for double_img in data:
    #     img= double_img[3:6].permute(1,2,0)
    #     (min_x, min_y), (max_x, max_y) = get_bounding_box(img)

    #     result = np.ascontiguousarray(img.numpy().astype('uint8'))
    #     plt.imshow(result)
    #     plt.show()
    #     cv.rectangle(result, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)        
    #     plt.imshow(result)
    #     plt.show()
    # print(calculate_phi((1,0), (0,1), (1,1), (1/2,0), (0, 1/2), (1/2, 1/2)))
    
    
    images, accel_arr, time_arr, delta_accel_arr, z_zero, z_dot_zero = torch.load("/home/tau/Video_Datasets/20Videos20Frames/2")
    # print(z_dot_zero / time_arr)
    phi_arr = (delta_accel_arr + z_zero + z_dot_zero) / z_zero
    # print(phi_arr)
    counter = 1
    for double_img in images:
        start_img= double_img[0:3].permute(1,2,0)
        (min_x_start, min_y_start), (max_x_start, max_y_start) = get_bounding_box(start_img)

        # set origin to center
        max_x_start -= len(start_img[0])/2
        min_x_start -= len(start_img[0])/2

        max_y_start -= len(start_img)/2
        min_y_start -= len(start_img)/2

        end_img= double_img[3:6].permute(1,2,0)
        (min_x_end, min_y_end), (max_x_end, max_y_end) = get_bounding_box(end_img)


        # set origin to center
        max_x_start -= len(start_img[0])/2
        min_x_start -= len(start_img[0])/2
        max_y_start -= len(start_img)/2
        min_y_start -= len(start_img)/2
        max_x_end -= len(end_img[0])/2
        min_x_end -= len(end_img[0])/2
        max_y_end -= len(end_img)/2
        min_y_end -= len(end_img)/2

        pred_phi = calculate_phi((min_x_start, min_y_start), (min_x_start, max_y_start), \
            (max_x_start, max_y_start), (min_x_end, min_y_end), (min_x_end, max_y_end), \
                (max_x_end, max_y_end))
        true_phi = phi_arr[counter]

        print('true phi: {:.4f} pred phi: {:.4f}'.format(true_phi, pred_phi))

        counter += 1
        # result = np.ascontiguousarray(img.numpy().astype('uint8'))
        # plt.imshow(result)
        # plt.show()
        # cv.rectangle(result, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)        
        # plt.imshow(result)
        # plt.show()
    

