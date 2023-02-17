from glob import glob
from matplotlib import pyplot as plt
import cv2
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

# from google.colab import drive, files
import os
import sys
import torch
import pytorch3d
import warnings

from data import SplineDataset, load_data_set
from model import Model
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import sys
import os
import math
import numpy as np
import gdown
import random
sys.path.append(os.path.abspath(''))
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LinearRegression
import datetime
import os

# access cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(device)

def least_squares(A:torch.tensor, b:torch.tensor) -> torch.tensor:
    if len(A.shape) == 1:
        A = A.reshape((A.shape[0], 1))
    if len(b.shape) == 1:
        b = b.reshape((b.shape[0], 1))
    x:torch.tensor = torch.matmul(torch.inverse((torch.matmul(torch.transpose(A, 0, 1), A))), torch.matmul(torch.transpose(A, 0, 1), b))

    return x

def get_k_from_phi(phi_vec:torch.tensor, time_vec:torch.tensor, delta_vec:torch.tensor,  z_g, second_index = 1):
    if len(phi_vec.shape) == 1:
        phi_vec = phi_vec.reshape((phi_vec.shape[0], 1))
    if len(time_vec.shape) == 1:
        time_vec = time_vec.reshape((time_vec.shape[0], 1))
    if len(delta_vec.shape) == 1:
        delta_vec = delta_vec.reshape((delta_vec.shape[0], 1))
    
    k = torch.tensor([1.0])

    for i in range(20000):
        k.requires_grad = True
        phi_offset = 1/k * phi_vec.clone().detach() + 1 - 1/k

        A1 = torch.cat(((phi_offset-1, -time_vec)), dim = 1)
        pos_and_vel1 = least_squares(A1, delta_vec)
        # print(phi_offset)
        A2 = torch.cat((phi_offset / phi_offset[second_index]-1, \
            -(time_vec-time_vec[second_index])), dim = 1)
        # print(A2)
        pos_and_vel2 = least_squares(A2, delta_vec)

        print('k: {:.4f} pos1: {:.4f} pos2: {:.4f}'.format(k.item(), pos_and_vel1[0].item(), pos_and_vel1[0].item()))
        # initial_pos_1 = pos_and_vel1[0]
        # initial_pos_2 = (phi_offset[0] / phi_offset[second_index]) * pos_and_vel2[0]

        z  = phi_offset*pos_and_vel1[0]
        z2 = (phi_offset / phi_offset[second_index])*pos_and_vel2[0]
        
        loss:torch.tensor = torch.norm(z2 - z)

        # loss:torch.tensor = (initial_pos_1 - initial_pos_2) ** 2
        loss.backward()

        # print(k.grad)
        plt.plot(z.detach())
        plt.plot(z2.detach())
        plt.plot(z_g.detach())
        plt.show()

        # print(torch.clone(k).detach())
        # print(k.grad)
        k = torch.clone(k).detach() - k.grad * 100.0
    
    adjusted_phi_vec = phi_vec.clone().detach() * k - k + 1
    print(adjusted_phi_vec)

    return adjusted_phi_vec

if __name__ == '__main__':
    # phi_vec = torch.tensor([1.0,2,3,4,5,2,3,5,2,3,4])
    # time_vec = torch.tensor([1.0,2,3,4,5,6,7,8,9,10,11])
    # delta_vec = (phi_vec - 1) * 123 + time_vec * 10 # torch.tensor([3,4,5,6,7,4,3,3,6,5,2.0])


    z0 = 1
    dz0= 1
    a = 10
    t = torch.range(0, 10)

    z = z0 + dz0 * t + 0.5 * a * t**2

    phistar = z / z[0]
    delta = 0.5 * a * t**2

    k_gt = 3.1459
    phi = k_gt * (phistar - 1) + 1


    get_k_from_phi(phi, t, delta, z)