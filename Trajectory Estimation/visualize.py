
import glob
import os
import string
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R
import pandas as pd
from scipy.interpolate import interp1d
import cv2 as cv
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df_path = '/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/reference_frame_index_0/data.csv'
    df = pd.read_csv(df_path)

    image_dir_path = '/home/tau/Desktop/monocular_data/dataset-corridor1_512_16/dso/cam0/undistorted_images'
    image_paths = []

    reference_index = 0

    # image_arr = []
    # for path in image_paths:
    #     image_arr.append(cv.imread(path))
    
    plt.ion()
    for index in range(len(os.listdir(image_dir_path))):
        image = cv.imread(image_dir_path + '/'+str(index) + '.png')
        # print(image_dir_path + '/{index}.png')
        if index >= reference_index:
            delta_pos = [df['x_delta'][index-reference_index], df['y_delta'][index-reference_index], df['z_delta'][index-reference_index]]
            delta_vels = [df['x_delta_vel'][index-reference_index], df['y_delta_vel'][index-reference_index], df['z_delta_vel'][index-reference_index]]
            accel = [df['x_accel'][index-reference_index], df['y_accel'][index-reference_index], df['z_accel'][index-reference_index]]
            rot_mat = R.from_quat((\
                    df['qx'][index-reference_index], \
                    df['qy'][index-reference_index], \
                    df['qz'][index-reference_index], \
                    df['qw'][index-reference_index]) \
                ).as_matrix()
            # print(rot_mat)
            
            fontScale = 0.5
            color = (255, 255, 255)
            thickness = 1
            
            plt.clf()
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.plot([0, rot_mat[0][0]/ (rot_mat[2][0]+2)], [0, -rot_mat[1][0]/ (rot_mat[2][0]+2)], 'r-')
            plt.plot([0, rot_mat[0][1]/ (rot_mat[2][1]+2)], [0, -rot_mat[1][1]/ (rot_mat[2][1]+2)], 'g-')
            plt.plot([0, rot_mat[0][2]/ (rot_mat[2][2]+2)], [0, -rot_mat[1][2]/ (rot_mat[2][2]+2)], 'b-')
            plt.plot([0, delta_vels[0]/ (delta_vels[2]+2) * 0.25], [0, -delta_vels[1]/ (delta_vels[2]+2) * 0.25], '-p')
            plt.plot()

            plt.pause(0.01)
            
            plt.show()

            cv.putText(image, 'x_delta:' + str(delta_pos[0])[0:4], (150,25), cv.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv.LINE_AA)
            cv.putText(image, 'y_delta:' + str(delta_pos[1])[0:4], (150,50), cv.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv.LINE_AA)
            cv.putText(image, 'z_delta:' + str(delta_pos[2])[0:4], (150,75), cv.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv.LINE_AA)

            # cv.putText(image, 'x_vel_delta:' + str(delta_vels[0])[0:4], (50,25), cv.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv.LINE_AA)
            # cv.putText(image, 'y_vel_delta:' + str(delta_vels[1])[0:4], (50,50), cv.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv.LINE_AA)
            # cv.putText(image, 'z_vel_delta:' + str(delta_vels[2])[0:4], (50,75), cv.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv.LINE_AA)
        
        cv.imshow('test', image)
        cv.waitKey(0)

    
    # x_vel_delta = interp1d()












