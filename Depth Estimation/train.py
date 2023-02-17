# from glob import glob
from matplotlib import pyplot as plt
# import cv2 
# import random
import numpy as np
# from numpy.linalg import inv
import torch
# from torch import Tensor
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import math
# from time import time
# from scipy.spatial.transform import Rotation as Rot
# import math
# from torch import scalar_tensor
# import time
# import copy
# import gc

from draw_bounding_box import calculate_phi,get_bounding_box

# from google.colab import drive, files
import os
import sys
import torch
# import pytorch3d
import warnings

from data import SplineDataset, load_data_set
from model import Model
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

# from tqdm import tqdm
# Util function for loading meshes
# from pytorch3d.io import load_objs_as_meshes, load_obj

# # Data structures and functions for rendering
# from pytorch3d.structures import Meshes
# # from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
# from pytorch3d.renderer import (
#     look_at_view_transform,
#     FoVPerspectiveCameras, 
#     PointLights, 
#     DirectionalLights, 
#     Materials, 
#     RasterizationSettings, 
#     MeshRenderer, 
#     MeshRasterizer,  
#     SoftPhongShader,
#     TexturesUV,
#     TexturesVertex
# )
import sys
import os
import math
import numpy as np
# import gdown
# import random
sys.path.append(os.path.abspath(''))
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
# from sklearn.linear_model import LinearRegression
import datetime
import os
import matplotlib
matplotlib.use('agg')
# access cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(device)

# custom loss function,
# the my_outputs is a tensor matrix with only 1 column and it represents the phi predictions from our NN
# deltas_and_times is a tensor matrix with 2 columns: the first one is double integral of acceleration, the second are the time values 
def custom_loss(phi_hat, deltas_and_times): # my_outputs are the phi output approximations, auxillary_info are the time and delta info
    # print("hi")
    # print(my_outputs.size())
    phi_hat.reshape(phi_hat.size()[0], 1)
    deltas = deltas_and_times[:,0]
    deltas = deltas.reshape(deltas.size()[0], 1) # delta is a single column of values
    times = deltas_and_times[:,1]
    times = times.reshape(times.shape[0], 1)
    
    # times = times.reshape(times.size()[0], 1) # times is a single column of values
  
    phi_and_time = torch.cat((torch.sub(phi_hat, 1), torch.multiply(times, -1)), 1) # make a matrix where first column is phi-1, second column is -time

    # solve the least squares for Z(0) and Z'(0)
    transpose = torch.transpose(phi_and_time, 0, 1)
    # print("transpose: ")
    # print(transpose)
    product = torch.matmul(transpose, phi_and_time) # 2 by 2 matrix
    # print("product: ")
    # print(product)   
    inverse = torch.inverse(product)
    # print("inverse")
    # print(inverse)
    Z_and_Z_vel = torch.matmul(torch.matmul(inverse, transpose), deltas) # first entry is estimated Z(0), second is estimated Z'(0)
    # print("Z_and_Z_vel")
    # print(Z_and_Z_vel)
    # print("Hi")
    # print(Z_and_Z_vel)

    # Z_and_Z_vel_actual = torch.tensor([[np.double(3.0)],[np.double(0.0)]]).to(device)
    # Z_vel = torch.tensor(Z_and_Z_vel[1])

    delta_accel_from_phi = torch.matmul(phi_and_time, Z_and_Z_vel)
    residues = torch.sub(delta_accel_from_phi, deltas) # difference between predicted delta values and true delta values
    residues = 1000*torch.norm(residues)**2 
    
    # print(Z_and_Z_vel)
    return residues, delta_accel_from_phi, deltas, Z_and_Z_vel[0], Z_and_Z_vel[1]  # returns the norm of the residue vector (ie square all the terms and add them together)

def least_squares(A:torch.tensor, b:torch.tensor) -> torch.tensor:
    if len(A.shape) == 1:
        A = A.reshape((A.shape[0], 1))
    if len(b.shape) == 1:
        b = b.reshape((b.shape[0], 1))
    x:torch.tensor = torch.matmul(torch.inverse((torch.matmul(torch.transpose(A, 0, 1), A))), torch.matmul(torch.transpose(A, 0, 1), b))

    return x



count = 0

def train(Training_Video_Num = 1000, Learning_rate = 1e-3, Frames = 100, \
    Epochs = 200, TrainingData = None, ValidationData = None, batch_size_train = 2, \
        batch_size_val = 2, writer=None, path=None):
    # set up directory for the run
    # path = '../deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    os.mkdir(path + 'validation_set')
    os.mkdir(path + 'training_set')
    os.mkdir(path + 'weights')
    # writer = SummaryWriter(log_dir = path + '/' + 'tensorboard_dir')

    try:
        # Training dataloader
        # if TrainingData is None:
        #     TrainingData = SplineDataset(Training_Video_Num, frames = Frames)
        # TrainLoader = DataLoader(TrainingData, batch_size_train)

        # actual training portion
        model = Model()
        model.load_state_dict(torch.load('/home/tau/deep_tau_runs/2022-08-08_23:21:42.001165/weights/model_weight95.hdf5'))
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=Learning_rate)

        # Validation dataloader
        if ValidationData is None:
            ValidationData = load_data_set('/home/tau/Video_Datasets/250Videos100Frames') # generate videos for validation
        ValidationLoader = DataLoader(ValidationData, batch_size_val)

        # set up tensorboard writer
        # if writer is None:
        #     writer = SummaryWriter(log_dir = path + '/' + 'tensorboard_dir')
        writer.add_text("training params:", 'lr: ' + str(Learning_rate) + '\nframes: ' + str(Frames) +'\nVideos: ' + str(Training_Video_Num)+'\nBatch size: ' + str(batch_size_train))
        print('Command to view tensorboard is:\ntensorboard --logdir ' + writer.log_dir)

        validation_file = open(path + "validation.txt","w")
        training_file = open(path + "training.txt","w")
        # global RMS_accuracy
        # RMS_accuracy = np.array([])
        # # training loop
        for epoch in range(Epochs):
            def run_batch_train(img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero,path,i,epoch) -> torch.tensor:
                batches_num = img_batches.shape[0]
                batch_loss = torch.tensor([0.0]).to(device)
                # print(z_zero.shape)
                # make a tensor in the shape (batches * (frames-1), 144, 144)
                img_pairs = torch.cat(tuple(img_batches[k] for k in range(batches_num)), dim = 0).float().to(device)
                # print(img_pairs.shape)
                # run all image pairs into the model
                phi_batch = model(img_pairs)
                # print(phi_batch.shape)
                # reshape tensor into shape (batches, frames-1)
                phi_batch = torch.cat(tuple(phi_batch[j * phi_batch.shape[0] // \
                    batches_num: (j + 1) * phi_batch.shape[0] // batches_num] for \
                        j in range(batches_num)), dim = 1).permute((1,0)).to(device)
                # print(phi_batch.shape)
                for j in range(batches_num): # for each batch in our batches
                    accel_arr = accel_batches[j].to(device)
                    time_arr = time_batches[j].to(device)
                    delta_accel_arr = delta_accel_batches[j].to(device)
                    deltas_and_time = torch.cat((torch.reshape(delta_accel_arr, (-1,1)), torch.reshape(time_arr, (-1,1))), 1).to(device)
                    
                    loss,delta_accel_from_phi,delta_accel_actual,predicted_depth, predicted_velocity \
                        = custom_loss((phi_batch[j])[:,None], deltas_and_time[1:])
                    global count
                    count += 1

                    actual_depth  = z_zero[j].cpu().numpy()
                    content = '{}: {}: loss {:.4f} z_gt {:.4f} z_predicted {:.4f}'.format(epoch, batches_num*i+j, loss.item()/1000, actual_depth, predicted_depth.data[0])

                    training_file.write(content)
                    training_file.write('\n')

                    if count %1 == 0:
                        z_dot_zero_tensor = (z_dot_zero[j]).to(device)
                        delta_accel_arr_gpu = delta_accel_arr.to(device)
                        z_zero_gpu = z_zero[j].to(device)
                        phi_actual = torch.tensor(1 + z_dot_zero_tensor/z_zero_gpu + delta_accel_arr_gpu/z_zero_gpu).to(device)

                        phi_hat = phi_batch[j].detach().cpu().numpy().flatten()
                        phi_gt = phi_actual.detach().cpu().numpy()

                        z0  = z_zero[j].detach().cpu().numpy()
                        dz0 = z_dot_zero_tensor.detach().cpu().numpy().flatten()
                        z0_predicted = predicted_depth.detach().cpu().numpy()
                        time_copy = time_arr.detach().cpu().numpy().flatten()
                        dz0_predicted = (predicted_velocity.detach().cpu().numpy().flatten())*(time_copy)

                        delta_accel_actual_numpy = delta_accel_actual.detach().cpu().numpy()
                        delta_accel_from_phi_numpy = delta_accel_from_phi.detach().cpu().numpy()

                        fig,axs =plt.subplots(nrows=4, ncols = 1, sharey = True,figsize = (10,15))
                        axs[3].get_shared_y_axes().remove(axs[3])
                        axs[0].plot(time_copy[1:], delta_accel_actual_numpy, label='actual_delta')
                        axs[0].plot(time_copy[1:],delta_accel_from_phi_numpy, label='predicted_delta')
                        axs[0].legend()
                        axs[1].plot(time_copy,(z0*(phi_gt-1)).flatten(), label='actual_(phi-1)*z(0)')
                        axs[1].plot(time_copy[1:], z0_predicted*(phi_hat-1), label='predicted_(phi-1)*z(0)')
                        axs[1].legend()
                        axs[2].plot(time_copy,(-dz0), label='-actual_dz0*t')
                        axs[2].plot(time_copy,(-dz0_predicted), label='-predicted_dz0*t')
                        axs[2].legend()
                        axs[3].plot(time_copy,phi_gt, label='phi_gt')
                        axs[3].plot(time_copy[1:],phi_hat, label='phi_hat')
                        axs[3].legend()

                        plt.savefig(path + "training_set/epoch_" + str(epoch) + "_train_number_" + str(batches_num*i+j) + "_.png",bbox_inches='tight')
                        plt.close(fig)
                        
                    batch_loss += loss
                # counter = 0 

                return batch_loss/batches_num
            
            def run_batch_val(img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero,path,i,epoch) -> torch.tensor:
                with torch.no_grad():
                    batches_num = img_batches.shape[0]
                    batch_loss = torch.tensor([0.0]).to(device)

                    # make a tensor in the shape (batches * (frames-1), 144, 144)
                    img_pairs = torch.cat(tuple(img_batches[k] for k in range(batches_num)), dim = 0).float().to(device)
                    
                    # run all image pairs into the model
                    phi_batch = model(img_pairs)

                    # reshape tensor into shape (batches, frames-1)
                    phi_batch = torch.cat(tuple(phi_batch[j * phi_batch.shape[0] // \
                        batches_num: (j + 1) * phi_batch.shape[0] // batches_num] for \
                            j in range(batches_num)), dim = 1).permute((1,0)).to(device)
                    
                    for j in range(batches_num): # for each batch in our batches
                        accel_arr = accel_batches[j].to(device)
                        time_arr = time_batches[j].to(device)
                        delta_accel_arr = delta_accel_batches[j].to(device)
                        deltas_and_time = torch.cat((torch.reshape(delta_accel_arr, (-1,1)), \
                            torch.reshape(time_arr, (-1,1))), 1).to(device)
                        
                        loss,delta_accel_from_phi,delta_accel_actual,predicted_depth, predicted_velocity \
                            = custom_loss((phi_batch[j])[:,None], deltas_and_time[1:])
                        global count
                        count += 1

                        phi_bounding_and_model = []
                        for l in range(5):
                            double_img = img_batches[j][len(img_batches[j]) // 5 * l + 1]
                            phi_bounding_box = run_batch_test(double_img[0:3], double_img[3:6])
                            phi_model = phi_batch[j][len(img_batches[j]) // 5 * l + 1]
                            phi_model = phi_model.cpu().item()
                            phi_bounding_and_model.append([phi_model, phi_bounding_box])
                        # print(phi_bounding_and_model)
                        phi_bounding_and_model = np.array([phi_bounding_and_model])
                        phi_from_model = phi_bounding_and_model[:,:,0] - 1
                        # print(phi_from_model.shape)
                        # print(phi_from_model)
                        phi_from_bounding = phi_bounding_and_model[:,:,1] - 1
                        # print(phi_from_bounding)
                        # print(phi_from_bounding.shape)
                        phi_from_bounding_transpose = phi_from_bounding.transpose()
                        # print(phi_from_model.shape)
                        # print(phi_from_model_transpose.shape)
                        # print(np.matmul(phi_from_model,phi_from_model_transpose))
                        # print(phi_from_bounding)
                        # print(np.matmul(phi_from_bounding,phi_from_bounding_transpose))
                        phi_from_bounding_inverse = np.linalg.inv(np.matmul(phi_from_bounding,phi_from_bounding_transpose))

                        k = np.matmul(phi_from_bounding_inverse,np.matmul(phi_from_model,phi_from_bounding_transpose))
                        # print(k.item())
                        actual_depth  = z_zero[j].cpu().numpy()
                        # print(predicted_depth.data[0].item())
                        # RMS_accuracy.append((actual_depth - predicted_depth.data[0].item()*k.item())**2)
                        # acc = math.sqrt(np.mean(RMS_accuracy))
                        content = '{}: {}: loss {:.4f} z_gt {:.4f} z_predicted {:.4f} k {:.4f} accuracy {}'.format(epoch, batches_num*i+j, loss.item()/1000, actual_depth, predicted_depth.data[0].item()*k.item(), k.item() , (actual_depth - predicted_depth.data[0].item()*k.item())/actual_depth * 100)

                        # print('RMS_accuracy = ' + str())
                        validation_file.write(content)
                        validation_file.write('\n')

                        if count %10 == 0:

                            z_dot_zero_tensor = (z_dot_zero[j]).to(device)
                            delta_accel_arr_gpu = delta_accel_arr.to(device)
                            z_zero_gpu = z_zero[j].to(device)
                            phi_actual = torch.tensor(1 + z_dot_zero_tensor/z_zero_gpu + delta_accel_arr_gpu/z_zero_gpu).to(device)

                            phi_hat = phi_batch[j].cpu().numpy().flatten()
                            phi_hat_adjusted = (phi_hat-1)/k.item() + 1
                            phi_gt = phi_actual.cpu().numpy()

                            z0  = z_zero[j].cpu().numpy()
                            dz0 = z_dot_zero_tensor.cpu().numpy().flatten()
                            z0_predicted_offset = predicted_depth.cpu().numpy()
                            z0_predicted = z0_predicted_offset*k.item()
                            time_copy = time_arr.cpu().numpy().flatten()
                            dz0_predicted = (predicted_velocity.cpu().numpy().flatten())*(time_copy)

                            delta_accel_actual_numpy = delta_accel_actual.cpu().numpy()
                            delta_accel_from_phi_numpy = delta_accel_from_phi.cpu().numpy()

                            fig,axs =plt.subplots(nrows=4, ncols = 1, sharey = True,figsize = (10,15))
                            axs[3].get_shared_y_axes().remove(axs[3])
                            axs[0].plot(time_copy[1:], delta_accel_actual_numpy, label='actual_delta')
                            axs[0].plot(time_copy[1:],delta_accel_from_phi_numpy, label='predicted_delta')
                            axs[0].legend()
                            axs[1].plot(time_copy,(z0*(phi_gt-1)).flatten(), label='actual_(phi-1)*z(0)')
                            axs[1].plot(time_copy[1:], z0_predicted_offset*(phi_hat-1), label='predicted_(phi-1)*z(0)')
                            axs[1].legend()
                            axs[2].plot(time_copy,(-dz0), label='-actual_dz0*t')
                            axs[2].plot(time_copy,(-dz0_predicted), label='-predicted_dz0*t')
                            axs[2].legend()
                            axs[3].plot(time_copy,phi_gt, label='phi_gt')
                            axs[3].plot(time_copy[1:],phi_hat, label='phi_hat')
                            axs[3].plot(time_copy[1:],phi_hat_adjusted, label='phi_hat adjusted, k=' + str(k.item()))
                            axs[3].legend()
                        
                            plt.savefig(path + "validation_set/epoch_" + str(epoch) + "_val_number_" + str(batches_num*i+j) + "_.png",bbox_inches='tight')
                            
                            # fig.clear()
                            plt.close(fig)
                            # axs.clear()
                            # plt.close(axs)
                            # plt.show()
                            # plt.clf()

                            # plt.figure.clear()
                            # plt.close()
                            # plt.cla()
                            # plt.clf()
                            # plt.close('all')
                            # gc.collect()

                        # print(count)
                        batch_loss += loss

                return batch_loss/batches_num
            
            def run_batch_test(img_reference, img_new):
                (min_x_start, min_y_start), (max_x_start, max_y_start) = get_bounding_box(img_reference)
                (min_x_end, min_y_end), (max_x_end, max_y_end) = get_bounding_box(img_new)
                
                max_x_start -= len(img_reference[0])/2
                min_x_start -= len(img_reference[0])/2
                max_y_start -= len(img_reference)/2
                min_y_start -= len(img_reference)/2
                max_x_end -= len(img_new[0])/2
                min_x_end -= len(img_new[0])/2
                max_y_end -= len(img_new)/2
                min_y_end -= len(img_new)/2

                pred_phi = calculate_phi((min_x_start, min_y_start), (min_x_start, max_y_start), \
                    (max_x_start, max_y_start), (min_x_end, min_y_end), (min_x_end, max_y_end), \
                        (max_x_end, max_y_end))


                return pred_phi

            
            sum_of_train_loss = 0
            sum_of_val_loss = 0
            
            global count 
            count = 0
            # One epoch of training loop
            # for i, data in enumerate(TrainLoader):
                
            #     img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero = data
            #     # print(img_batches.shape)
            #     batch_loss = run_batch_train(img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero,path,i,epoch)\
            #          * batch_size_train
            #     batch_loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()

            #     sum_of_train_loss += batch_loss.item()

            
            count = 0
            # One epoch of validation loop
            for i, data in enumerate(ValidationLoader):
                # print(i)
                img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero = data
                
                batch_loss = run_batch_val(img_batches, accel_batches, time_batches, delta_accel_batches, z_zero, z_dot_zero,path,i,epoch)\
                     * batch_size_val

                sum_of_val_loss += batch_loss.item()

            if epoch%5 == 0:
                torch.save(model.state_dict(), path + "weights/model_weight" + str(epoch) + ".hdf5")

            # print('epoch: {} avg_train_loss: {:.4f} avg_val_loss: {:.4f}'\
            #     .format(epoch, sum_of_train_loss / (len(TrainLoader) * batch_size_train * 1000), \
            #         sum_of_val_loss / (len(ValidationLoader) * batch_size_val * 1000)))
            # writer.add_scalars('Losses during training', {'avg training loss':sum_of_train_loss / (len(TrainLoader) * batch_size_train * 1000),
            #                             'avg validation loss':sum_of_val_loss / (len(ValidationLoader) * batch_size_val * 1000)}, epoch)
            sum_of_train_loss = 0
            sum_of_val_loss = 0

    except RuntimeError as e:
        print(e)
        # python3 myscript.py 2>&1 | tee output.txt
        torch.cuda.empty_cache()
    
    writer.close()
    torch.cuda.empty_cache()
    validation_file.close()
    training_file.close()

if __name__ == '__main__':

    # TrainingData = SplineDataset(5, frames = 20)
    # data = TrainingData.__getitem__(0)[0]
    # for i in range(len(data)):
    #     print(len(data))
    #     plt.imshow(data[i][0:3].permute(1,2,0).numpy())
    #     plt.imshow(data[i][3:6].permute(1,2,0).numpy())
    #     plt.show()

    path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    os.mkdir(path)
    writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set("/home/tau/Video_Datasets/1000Videos200Frames/"),batch_size_train= 1, writer = writer, path=path, Frames = 200, Training_Video_Num=1000, Epochs=100)
    train(batch_size_train= 1, writer = writer, path=path, Frames = 200, Training_Video_Num=1000, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/1000Videos150Frames/"), batch_size_train=1,writer=writer, path = path, Frames = 150, Training_Video_Num=1000,Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/750Videos150Frames/"), batch_size_train=1, writer=writer, path = path, Frames = 150, Training_Video_Num=750,Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/500Videos150Frames/"), batch_size_train=1, writer=writer, path = path,Frames = 150, Training_Video_Num=500, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/250Videos150Frames/"), batch_size_train=1, writer=writer, path = path , Frames = 150, Training_Video_Num=250, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/100Videos150Frames/"), batch_size_train=1, writer=writer, path = path, Frames = 150, Training_Video_Num=100, Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set ("/home/tau/Video_Datasets/50Videos150Frames/"), batch_size_train=1, writer=writer, path = path, Frames = 150, Training_Video_Num=50,Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')    
    # train(TrainingData = load_data_set ("/home/tau/Video_Datasets/20Videos150Frames/"), batch_size_train=1, writer=writer, path = path, Frames = 150, Training_Video_Num=20, Epochs=200)

    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set ("/home/tau/Video_Datasets/1000Videos100Frames/"), batch_size_train=2, writer=writer, path = path, Frames = 100, Training_Video_Num=1000, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/500Videos100Frames/"), batch_size_train=2, writer=writer, path = path , Frames = 100, Training_Video_Num=500, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/250Videos100Frames/"), batch_size_train=2, writer=writer, path = path , Frames = 100, Training_Video_Num=250, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/100Videos100Frames/"), batch_size_train=2, writer=writer, path = path, Frames = 100, Training_Video_Num=100, Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/50Videos100Frames/"), batch_size_train=2, writer=writer, path = path, Frames = 100, Training_Video_Num=50, Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')    
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/20Videos100Frames/"), batch_size_train=2, writer=writer, path = path, Frames = 100, Training_Video_Num=20, Epochs=200)

    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/1000Videos50Frames/"), batch_size_train=4, writer=writer, path = path, Frames = 50, Training_Video_Num=1000, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/750Videos50Frames/"), batch_size_train=4, writer=writer, path = path, Frames = 50, Training_Video_Num=750, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/500Videos50Frames/"), batch_size_train=4, writer=writer, path = path, Frames = 50, Training_Video_Num=500, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/250Videos50Frames/"), batch_size_train=4, writer=writer, path = path, Frames = 50, Training_Video_Num=250, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/100Videos50Frames/"), batch_size_train=4, writer=writer, path = path, Frames = 50, Training_Video_Num=100, Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set("/home/tau/Video_Datasets/50Videos50Frames/"), batch_size_train=4, writer=writer, path = path, Frames = 50, Training_Video_Num=50, Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')    
    # train(TrainingData = load_data_set("/home/tau/Video_Datasets/20Videos50Frames/"), batch_size_train=4, writer=writer, path = path, Frames = 50, Training_Video_Num=20, Epochs=200)

    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/1000Videos20Frames/"), batch_size_train=8, writer=writer, path = path,Frames = 20, Training_Video_Num=1000, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/750Videos20Frames/"), batch_size_train=8, writer=writer, path = path,Frames = 20, Training_Video_Num=750, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/500Videos20Frames/"), batch_size_train=8, writer=writer, path = path, Frames = 20, Training_Video_Num=500, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/250Videos20Frames/"), batch_size_train=8, writer=writer, path = path,Frames = 20, Training_Video_Num=250, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/100Videos20Frames/"), batch_size_train=8, writer=writer, path = path,Frames = 20, Training_Video_Num=100, Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set("/home/tau/Video_Datasets/50Videos20Frames/"), batch_size_train=8, writer=writer, path = path,Frames = 20, Training_Video_Num=50, Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')    
    # train(TrainingData = load_data_set("/home/tau/Video_Datasets/20Videos20Frames/"), batch_size_train=8, writer=writer, path = path,Frames = 20, Training_Video_Num=20, Epochs=200)

    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/1000Videos10Frames/"), batch_size_train=16,writer = writer, path=path,Frames = 10, Training_Video_Num=1000, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/750Videos10Frames/"), batch_size_train=16,writer = writer, path=path,Frames = 10, Training_Video_Num=750, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/500Videos10Frames/"), batch_size_train=16,writer = writer, path=path,Frames = 10, Training_Video_Num=500, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/250Videos10Frames/"), batch_size_train=16,writer = writer, path=path,Frames = 10, Training_Video_Num=250, Epochs=100)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set( "/home/tau/Video_Datasets/100Videos10Frames/"), batch_size_train=16,writer = writer, path=path,Frames = 10, Training_Video_Num=100, Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set("/home/tau/Video_Datasets/50Videos10Frames/"), batch_size_train=16,writer = writer, path=path,Frames = 10, Training_Video_Num=50, Epochs=200)
    
    # path = '/home/tau/deep_tau_runs/' + str(datetime.datetime.now()).replace(' ', '_') + '/'
    # os.mkdir(path)
    # os.mkdir(path + 'tensorboard_dir')
    # writer = SummaryWriter(path + '/'+ 'tensorboard_dir')
    # train(TrainingData = load_data_set("/home/tau/Video_Datasets/20Videos10Frames/"), batch_size_train=16,writer = writer, path=path,Frames = 10, Training_Video_Num=20, Epochs=200)    
