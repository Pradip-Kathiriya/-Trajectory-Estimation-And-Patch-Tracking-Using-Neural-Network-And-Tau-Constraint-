from glob import glob
from matplotlib import pyplot as plt
import cv2
import random
import numpy as np
from numpy.linalg import inv
import torch
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



# access cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Function takes position, velocity, and acceleration functions and returns images and array of information
def render_from_func(physical_scale, camera_pos_func, camera_vel_func, \
    camera_accel_func, time_step = 1/10, max_time = 0.1):
  raster_settings = RasterizationSettings(
    image_size=128, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
  )
  # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
  # -z direction. 
  lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

  # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
  # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
  # apply the Phong lighting model
  renderer = MeshRenderer(
      rasterizer=MeshRasterizer(
          raster_settings=raster_settings
      ),
      shader=SoftPhongShader(
          device=device, 
          lights=lights
      )
  )
  # Load mesh
  DATA_DIR = "./data"
  obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

  # # Load obj file
  mesh = load_objs_as_meshes([obj_filename], device=device) # get a mesh object

  verts = mesh.verts_packed()
  N = verts.shape[0]
  center = verts.mean(0)
  scale = max((verts - center).abs().max(0)[0])
  # mesh.offset_verts_(-center)
  mesh.scale_verts_((1.0 / float(scale))*physical_scale)

  img_num = round(max_time / time_step)
  meshes = mesh.extend(img_num)

  T = torch.tensor([camera_pos_func(i * time_step) for i in range(img_num)]) # an array of positions of your camera
  R = torch.tensor([[[1,0,0],[0,1,0],[0,0,1]] for i in range(img_num)]) # rotation matrix, held constant
  
  cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
  img_arr = renderer(meshes, lights=lights, cameras=cameras).cpu().numpy()
  img_arr = np.uint16(img_arr * 255)
  
  # create array of camera positions, times, accelerations, and delta of acceleration
  camera_pos = [] # z direction
  velo_arr = []
  accel_arr = [] # z direction
  time_arr = []
  delta_accel_arr = [] # z direction

  for i in range(img_num):
      time = i * time_step
      camera_pos.append(camera_pos_func(time))
      velo_arr.append(camera_vel_func(0)[2]*time)
      accel_arr.append(camera_accel_func(time)[2])
      time_arr.append(time)
      delta_accel_arr.append(camera_pos[i][2] - camera_pos[0][2] - camera_vel_func(0)[2] * time)
      # print(camera_pos[i])

  camera_pos = np.array(camera_pos)
  z_zero = camera_pos[0][2]
  z_dot_zero = velo_arr # column vector with 99 rows
  img_arr_new = []
  # img_arr = np.array(img_arr, dtype=float)
  for i in range(len(img_arr)):
    image = np.uint8(img_arr[i])
    R,G,B,Alpha = cv2.split(image)
    image = cv2.merge((R,G,B))
    img_arr_new.append(np.array(image))
  
  img_arr = np.array(img_arr_new)

  accel_arr = np.array(accel_arr, dtype=float)
  time_arr = np.array(time_arr, dtype=float)
  delta_accel_arr = np.array(delta_accel_arr)

  return [img_arr, accel_arr, time_arr, delta_accel_arr, z_zero, z_dot_zero]

# Function takes an array of points and corresponding velocities and accelerations
def render_from_points(physical_scale, camera_pos, camera_vel, camera_accel, time_arr):

  raster_settings = RasterizationSettings(
    image_size=128, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
  )
  # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
  # -z direction. 
  lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

  # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
  # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
  # apply the Phong lighting model
  renderer = MeshRenderer(
      rasterizer=MeshRasterizer(
          raster_settings=raster_settings
      ),
      shader=SoftPhongShader(
          device=device, 
          lights=lights
      )
  )
  # Load mesh
  DATA_DIR = "./data"
  obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

  # # Load obj file
  mesh = load_objs_as_meshes([obj_filename], device=device) # get a mesh object

  verts = mesh.verts_packed()
  N = verts.shape[0]
  center = verts.mean(0)
  scale = max((verts - center).abs().max(0)[0])
  # mesh.offset_verts_(-center)
  mesh.scale_verts_((1.0 / float(scale))*physical_scale)

  img_num = len(camera_pos)
  
  images_rendered = 0

  img_arr = []
  while images_rendered < img_num:
    # print(images_rendered)
    images_to_render = min(100, img_num - images_rendered)
    meshes = mesh.extend(images_to_render)
    T = torch.tensor(camera_pos[images_rendered:images_rendered + images_to_render]) # an array of positions of your camera
    R = torch.tensor([[[1,0,0],[0,1,0],[0,0,1]] for i in range(images_rendered, images_rendered + images_to_render)]) # rotation matrix, held constant
    
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    img_arr.append(np.uint16(renderer(meshes, lights=lights, cameras=cameras).detach().cpu().numpy() * 255))
    images_rendered += images_to_render
  img_arr = np.concatenate(tuple(img_arr)) 
  # create array of camera positions, times, accelerations, and delta of acceleration
  delta_accel_arr = [] # z direction
  velo_arr = []

  for i in range(img_num):
    time = time_arr[i]
    delta_accel_arr.append(camera_pos[i][2] - camera_pos[0][2] - camera_vel[0][2] * time)
    velo_arr.append(camera_vel[0][2]*time)
    # print(camera_pos[i])

  camera_pos = np.array(camera_pos)
  z_zero = camera_pos[0][2]
  z_dot_zero = velo_arr # column vector with 99 rows
  img_arr_new = []
  # img_arr = np.array(img_arr, dtype=float)
  for i in range(len(img_arr)):
    image = np.uint8(img_arr[i])
    R,G,B,Alpha = cv2.split(image)
    image = cv2.merge((R,G,B))
    img_arr_new.append(np.array(image))
  
  img_arr = np.array(img_arr_new)
  z_dot_zero = np.array(z_dot_zero)
  accel_arr = np.array(camera_accel, dtype=float)
  time_arr = np.array(time_arr, dtype=float)
  delta_accel_arr = np.array(delta_accel_arr)

  return [img_arr, accel_arr, time_arr, delta_accel_arr, z_zero, z_dot_zero]

def fit_cubic_spline_multi_var_pair(r_0_vec, r_1_vec, v_0_vec, v_1_vec, t_0, t_1):
  def fit_cubic_spline_single_dim_pair(r_0, r_1, v_0, v_1, t_0, t_1):
    # all inputs should be decimals
    coefficients = np.matmul(np.linalg.inv(np.array([
                            [t_0**3, t_0**2, t_0, 1],
                            [t_1**3, t_1**2, t_1, 1],
                            [3*t_0**2, 2*t_0, 1, 0],
                            [3*t_1**2, 2*t_1, 1, 0]])), np.array([[r_0], [r_1], [v_0], [v_1]]))
    coefficients = np.reshape(coefficients, 4)
    return coefficients

  
  coefficients = []
  dims = len(r_0_vec)
  for i in range(dims):
    fit_cubic_spline_single_dim_pair(r_0_vec[i], r_1_vec[i], v_0_vec[i], v_1_vec[i], t_0, t_1)
    coefficients.append(fit_cubic_spline_single_dim_pair(r_0_vec[i], r_1_vec[i], v_0_vec[i], v_1_vec[i], t_0, t_1))
  

  def spline_piece_pos(time):
    our_coefficients = coefficients
    our_dims = dims
    output = np.zeros(our_dims)
    for i in range(our_dims):
      output[i] = our_coefficients[i][0] * time**3 + our_coefficients[i][1] * time**2 + our_coefficients[i][2] * time + our_coefficients[i][3]
    return output

  def spline_piece_vel(time):
    our_coefficients = coefficients
    our_dims = dims
    output = np.zeros(our_dims)
    for i in range(our_dims):
      output[i] = our_coefficients[i][0] * 3 * time**2 + 2 * our_coefficients[i][1] * time + our_coefficients[i][2]
    return output

  def spline_piece_accel(time):
    our_coefficients = coefficients
    our_dims = dims
    output = np.zeros(our_dims)
    for i in range(our_dims):
      output[i] = our_coefficients[i][0] * 6 * time + 2 * our_coefficients[i][1]
    return output

  return (spline_piece_pos, spline_piece_vel, spline_piece_accel)

def fit_cubic_spline_through_sequence(points_list, vel_list, time_list):
    
  num_points = len(points_list)
  spline_pieces = []
  for i in range(num_points-1):
    spline_pieces.append(fit_cubic_spline_multi_var_pair(points_list[i], points_list[i + 1], vel_list[i], vel_list[i + 1], time_list[i], time_list[i + 1]))

  def full_spline_pos(time): 
    our_spline_pieces = spline_pieces
    our_t_0 = time_list[0]
    our_t_final = time_list[-1]
    our_time_list = time_list
    pos_func = None
    vel_func = None
    accel_func = None
    

    if our_t_0 <= time and our_t_final >= time:
      index = 0
      for index in range(len(time_list) - 1):
        if time_list[index] <= time and time_list[index+1] >= time:
          break
      (pos_func, vel_func, accel_func) = spline_pieces[index]
    elif time < our_t_0:
      (pos_func, vel_func, accel_func) = spline_pieces[0]
    else:
      (pos_func, vel_func, accel_func) = spline_pieces[-1]

    return pos_func(time)


  def full_spline_vel(time): 
    our_spline_pieces = spline_pieces
    our_t_0 = time_list[0]
    our_t_final = time_list[-1]
    our_time_list = time_list
    pos_func = None
    vel_func = None
    accel_func = None
    
    if our_t_0 <= time and our_t_final >= time:
      index = 0
      for index in range(len(time_list) - 1):
        if time_list[index] <= time and time_list[index+1] >= time:
          break
      (pos_func, vel_func, accel_func) = spline_pieces[index]
    elif time < our_t_0:
      (pos_func, vel_func, accel_func) = spline_pieces[0]
    else:
      (pos_func, vel_func, accel_func) = spline_pieces[-1]

    return vel_func(time)


  def full_spline_accel(time): 
    our_spline_pieces = spline_pieces
    our_t_0 = time_list[0]
    our_t_final = time_list[-1]
    our_time_list = time_list
    pos_func = None
    vel_func = None
    accel_func = None
    
    if our_t_0 <= time and our_t_final >= time:

      index = 0
      for index in range(len(time_list) - 1):
        if time_list[index] <= time and time_list[index+1] >= time:
          break
      (pos_func, vel_func, accel_func) = spline_pieces[index]
    elif time < our_t_0:
      (pos_func, vel_func, accel_func) = spline_pieces[0]
    else:
      (pos_func, vel_func, accel_func) = spline_pieces[-1]

    return accel_func(time)

  return (full_spline_pos, full_spline_vel, full_spline_accel)


random_values_from_0_to_1 = np.random.rand(10000)
random_time_index = 0
def get_spline_video(num = 5, xy_bound= 0.45, z_lower = 1.5, z_upper = 5, frames = 100):
  global random_values_from_0_to_1
  global random_time_index
  
  
  points = np.array([[random.uniform(-xy_bound, xy_bound), \
    random.uniform(-xy_bound, xy_bound), random.uniform(z_lower, z_upper)] for i in range(num)])
  vel = np.array([[random.uniform(-2*xy_bound, 2*xy_bound), \
    random.uniform(-2*xy_bound, 2*xy_bound), random.uniform(-2*z_lower, 2*z_upper)] for i in range(num)])
  time = np.array(range(num))

  # print(points)
  # ax.scatter3D(points[:,0], points[:,1], points[:,2])

  spline_pos_func, spline_vel_func, spline_accel_func = fit_cubic_spline_through_sequence(points, vel, time)

  spline_points = []
  spline_vel = []
  spline_accel = []
  spline_time = []

  time_initial = time[0]
  time_final = time[-1]

  counter = 0
  while counter < frames:
    rand_time = None
    if random_time_index == len(random_values_from_0_to_1):
      random_values_from_0_to_1 = np.random.rand(10000)
      random_time_index = 0

    rand_time = (time_final - time_initial) * random_values_from_0_to_1[random_time_index] + time_initial
    random_time_index += 1
    # print(rand_time)
    rand_pos = spline_pos_func(rand_time)

    if rand_pos[0] < xy_bound and rand_pos[0] > -xy_bound and rand_pos[1] < xy_bound \
        and rand_pos[1] >-xy_bound and rand_pos[2] < z_upper and rand_pos[2] > z_lower:
      counter += 1
      spline_time.append(rand_time)
      # print((time[-1] - time[0]) / num_points * i + time[0])
    # print(str((time[1] - time[0]) / num_points * i + time[0]) + ", " + str(spline_points[-1]))
  spline_time.sort()


  for i in range(len(spline_time)):
      spline_points.append(spline_pos_func(spline_time[i]))
      spline_vel.append(spline_vel_func(spline_time[i]))
      spline_accel.append(spline_accel_func(spline_time[i]))
  spline_points = np.array(spline_points)
  return render_from_points(random.uniform(0.3,1), spline_points, spline_vel, spline_accel, spline_time)


# This custom dataset class outputs all the information needed to train the model. 
class SplineDataset(Dataset):
    def __init__(self, data_size, frames = 100):
      global random_values_from_0_to_1
      global random_time_index
      self.size = data_size
      self.data = []
      
      random_values_from_0_to_1 = np.random.rand(data_size * 20)
      random_time_index = 0
      
      for i in range(data_size): 
        if i % 250 == 0:
          print('generated ' + str(i) + ' videos')
        img_arr, accel_arr, time_arr, delta_accel_arr, z_zero, z_dot_zero \
            = (get_spline_video(num=8, frames = frames))

        img_arr = torch.tensor(img_arr)
        arr_of_doubles = []

        for k in range(1, len(img_arr)):
            
            double_img = torch.cat((img_arr[0], img_arr[k]),2)
            double_img = double_img.permute(2,0,1)[None,:]
            # plt.imshow(double_img[i][0:3].permute(1,2,0).numpy())
            # plt.imshow(double_img[i][3:6].permute(1,2,0).numpy())
            # plt.show()

            arr_of_doubles.append(double_img)


        tensor_of_doubles = torch.cat(tuple(double for double in arr_of_doubles), 0)

        self.data.append((tensor_of_doubles, accel_arr, time_arr, delta_accel_arr, z_zero, z_dot_zero))


    def __len__(self):
        return self.size
    
    # Note: if you want the same video every time, might be best to put
    # the code in the __init__ func so it doesn't regenerate the 
    # video turning every call of __getitem__
    
    def __getitem__(self, index):
        return self.data[index]


class DirectoryLoadedDataset(Dataset):
  
  def __init__(self, dir_path):
    num_files = len(os.listdir(dir_path))
    self.data = [None for i in range(num_files)]
    if dir_path[-1] != '/':
      dir_path += '/'
    counter = 0
    for file in os.listdir(dir_path):
      # print(file)
      self.data[counter] = torch.load(dir_path + file)
      counter += 1
    
  def __len__(self):
    return len(self.data)
    
  def __getitem__(self, index):
    return self.data[index]


def save_data_set(dataset:Dataset, directory_name:str) -> None:
  try:
    os.mkdir(directory_name)
  finally:
    for i in range(len(dataset)):
      if directory_name[-1] == '/':
        path = directory_name + str(i)
      else:
        path = directory_name + '/' + str(i)
      torch.save(dataset.__getitem__(i), path)


# directory must only contain tensor files, no directories
def load_data_set(directory_name:str) -> Dataset:
  print('loading data from ' + directory_name)
  return DirectoryLoadedDataset(directory_name)


if __name__ == '__main__':
  
  # save_data_set(SplineDataset(1000, frames = 200), "../Video_Datasets/1000Videos200Frames/")
  save_data_set(SplineDataset(1000, frames = 150), "../Video_Datasets/1000Videos150Frames/")
  save_data_set(SplineDataset(750, frames = 150), "../Video_Datasets/750Videos150Frames/")
  save_data_set(SplineDataset(500, frames = 150), "../Video_Datasets/500Videos150Frames/")
  save_data_set(SplineDataset(250, frames = 150), "../Video_Datasets/250Videos150Frames/")
  save_data_set(SplineDataset(100, frames = 150), "../Video_Datasets/100Videos150Frames/")
  save_data_set(SplineDataset(50, frames = 150), "../Video_Datasets/50Videos150Frames/")
  save_data_set(SplineDataset(20, frames = 150), "../Video_Datasets/20Videos150Frames/")

  save_data_set(SplineDataset(1000, frames = 100), "../Video_Datasets/1000Videos100Frames/")
  save_data_set(SplineDataset(750, frames = 100), "../Video_Datasets/750Videos100Frames/")
  save_data_set(SplineDataset(500, frames = 100), "../Video_Datasets/500Videos100Frames/")
  save_data_set(SplineDataset(250, frames = 100), "../Video_Datasets/250Videos100Frames/")
  save_data_set(SplineDataset(100, frames = 100), "../Video_Datasets/100Videos100Frames/")
  save_data_set(SplineDataset(50, frames = 100), "../Video_Datasets/50Videos100Frames/")
  save_data_set(SplineDataset(20, frames = 100), "../Video_Datasets/20Videos100Frames/")

  save_data_set(SplineDataset(1000, frames = 50), "../Video_Datasets/1000Videos50Frames/")
  save_data_set(SplineDataset(750, frames = 50), "../Video_Datasets/750Videos50Frames/")
  save_data_set(SplineDataset(500, frames = 50), "../Video_Datasets/500Videos50Frames/")
  save_data_set(SplineDataset(250, frames = 50), "../Video_Datasets/250Videos50Frames/")
  save_data_set(SplineDataset(100, frames = 50), "../Video_Datasets/100Videos50Frames/")
  save_data_set(SplineDataset(50, frames = 50), "../Video_Datasets/50Videos50Frames/")
  save_data_set(SplineDataset(20, frames = 50), "../Video_Datasets/20Videos50Frames/")

  # save_data_set(SplineDataset(1000, frames = 20), "../Video_Datasets/1000Videos20Frames/")
  # save_data_set(SplineDataset(750, frames = 20), "../Video_Datasets/750Videos20Frames/")
  # save_data_set(SplineDataset(500, frames = 20), "../Video_Datasets/500Videos20Frames/")
  # save_data_set(SplineDataset(250, frames = 20), "../Video_Datasets/250Videos20Frames/")
  # save_data_set(SplineDataset(100, frames = 20), "../Video_Datasets/100Videos20Frames/")
  # save_data_set(SplineDataset(50, frames = 20), "../Video_Datasets/50Videos20Frames/")
  # save_data_set(SplineDataset(20, frames = 20), "../Video_Datasets/20Videos20Frames/")

  save_data_set(SplineDataset(1000, frames = 10), "../Video_Datasets/1000Videos10Frames/")
  save_data_set(SplineDataset(750, frames = 10), "../Video_Datasets/750Videos10Frames/")
  save_data_set(SplineDataset(500, frames = 10), "../Video_Datasets/500Videos10Frames/")
  save_data_set(SplineDataset(250, frames = 10), "../Video_Datasets/250Videos10Frames/")
  save_data_set(SplineDataset(100, frames = 10), "../Video_Datasets/100Videos10Frames/")
  save_data_set(SplineDataset(50, frames = 10), "../Video_Datasets/50Videos10Frames/")
  save_data_set(SplineDataset(20, frames = 10), "../Video_Datasets/20Videos10Frames/")

