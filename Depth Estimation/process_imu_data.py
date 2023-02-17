from array import array
import os
import string
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R
import pandas as pd
import interpolate

def process(path:string):
    mcap_start_time = pd.read_csv(path + '/mav0/mocap0/data.csv')['#timestamp [ns]'][0]
    frame_start_time = pd.read_csv(path+'/mav0/cam0/data.csv')['#timestamp [ns]'][0]
    imu_start_time = pd.read_csv(path + '/mav0/imu0/data.csv')['#timestamp [ns]'][0]
    

    start_time = max(mcap_start_time, frame_start_time, imu_start_time)


    vicon_orientations = pd.read_csv(path + '/dso/gt_imu.csv')[['qx', 'qy', 'qz', 'qw']]
    
    vicon_interp = Slerp(pd.read_csv(path + '/dso/gt_imu.csv')['# timestamp[ns]'], R.from_quat(vicon_orientations))

    # print(start_time)
    # print(vicon_interp(start_time).as_quat())
    
    start_orientation_mat = vicon_interp(start_time).as_matrix()


    imu_rot_interp = interpolate.get_interpolation(path)
    imu_data = pd.read_csv(path + '/mav0/imu0/data.csv')
    imu_time_df = imu_data['#timestamp [ns]']
    a_x_imu_df = imu_data['a_RS_S_x [m s^-2]']
    a_y_imu_df = imu_data['a_RS_S_y [m s^-2]']
    a_z_imu_df = imu_data['a_RS_S_z [m s^-2]']

    new_df = pd.DataFrame({'time':[], 'a_x_world':[], 'a_y_world':[], 'a_z_world':[], \
        'qw_imu_to_world':[], 'qx_imu_to_world':[], 'qy_imu_to_world':[],'qz_imu_to_world':[]})
    # print(new_df)
    
    for i in range(len(imu_time_df)):
        imu_to_world_mat = start_orientation_mat  @ \
            np.linalg.inv(imu_rot_interp(start_time).as_matrix()) \
                @ imu_rot_interp(imu_time_df[i]).as_matrix()
        # print(world_to_imu_mat)
        # print(np.linalg.inv(world_to_imu_mat))
        
        
        accel_world = imu_to_world_mat @ np.array([[a_x_imu_df[i]], [a_y_imu_df[i]], [a_z_imu_df[i]]])
        accel_world[2] -= 9.80665
        # accel_in_world = np.linalg.inv(world_to_imu_mat) @ np.array([[a_x_imu_df[i]], [a_y_imu_df[i]], [a_z_imu_df[i] - 9.80665]])
        # # print(accel_in_world[0], accel_in_world[1], accel_in_world[2])

        # print(accel_world[0], accel_world[1], accel_world[2])

        imu_to_world_rot = R.from_matrix(imu_to_world_mat).as_quat()
        new_df.loc[len(new_df.index)] = [imu_time_df[i], accel_world[0][0], accel_world[1][0], accel_world[2][0], imu_to_world_rot[0], imu_to_world_rot[1], imu_to_world_rot[2], imu_to_world_rot[3]]
        # print(new_df)
        print(i+1, '/', len(imu_time_df))
    os.makedirs(path + '/correct_imu_data', exist_ok=True)
    new_df.to_csv(path + '/correct_imu_data/data.csv')


if __name__ == '__main__':
    process('/home/tau/Desktop/monocular_data/dataset-corridor1_512_16')