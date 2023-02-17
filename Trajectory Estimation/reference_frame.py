import os
import string
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R
import pandas as pd
from scipy.interpolate import interp1d

def get_info_from_frame(index:int, info_path:str, write_path:str):
    if info_path[-1] == '/':
        info_path = info_path[:-1]
    if write_path[-1] == '/':
        write_path = write_path[:-1]
 
    processed_imu_df = pd.read_csv(info_path + '/correct_imu_data/data.csv')
    frame_to_time_df = pd.read_csv(info_path + '/mav0/cam0/data.csv')

    new_df = pd.DataFrame({'frame_new_index':[], 'frame_old_index':[], 'time_stamp':[], 'original_time_stamp':None,\
        'x_accel':[], 'y_accel':[], 'z_accel':[], 'x_delta_vel':[], 'y_delta_vel':[], 'z_delta_vel':[], 'x_delta':[], 'y_delta':[], 'z_delta':[]})

    imu_orientation_interp = Slerp(processed_imu_df['time'],\
        R.from_quat(\
            processed_imu_df[['qx_imu_to_world', 'qy_imu_to_world', 'qz_imu_to_world', 'qw_imu_to_world']])\
        )
    
    
    new_reference_time = frame_to_time_df['#timestamp [ns]'][index]
    rot_mat_from_new_reference_to_old_reference = imu_orientation_interp(new_reference_time).as_matrix()


    for i in range(index, len(frame_to_time_df)):
        current_frame_time = frame_to_time_df['#timestamp [ns]'][i]
        
        row = {'frame_new_index':None, 'frame_old_index':None, 'time_stamp':None, 'original_time_stamp':current_frame_time,\
        'x_accel':[], 'y_accel':[], 'z_accel':[], 'x_delta_vel': None, 'y_delta_vel': None, 'z_delta_vel': None,'x_delta':[], 'y_delta':[], 'z_delta':[], 'qx':None, 'qy':None, 'qz':None, 'qw':None}
        row['frame_new_index'] = int(i-index)
        row['frame_old_index'] = int(i)
        row['time_stamp'] = (frame_to_time_df['#timestamp [ns]'][i] \
            - frame_to_time_df['#timestamp [ns]'][index]) / 10.0**9
        
        x_accel_world_interp = interp1d(processed_imu_df['time'], processed_imu_df['a_x_world'])
        y_accel_world_interp = interp1d(processed_imu_df['time'], processed_imu_df['a_y_world'])
        z_accel_world_interp = interp1d(processed_imu_df['time'], processed_imu_df['a_z_world'])

        adjusted_accel = rot_mat_from_new_reference_to_old_reference.T \
            @ np.vstack((x_accel_world_interp(current_frame_time), y_accel_world_interp(current_frame_time), z_accel_world_interp(current_frame_time)))
        
        
        row['x_accel'] = adjusted_accel[0]
        row['y_accel'] = adjusted_accel[1]
        row['z_accel'] = adjusted_accel[2]

        if i != index:
            row['x_delta_vel'] = new_df['x_accel'][i-index-1] * (row['time_stamp'] - new_df['time_stamp'][i-index-1]) + new_df['x_delta_vel'][i-index - 1]
            row['y_delta_vel'] = new_df['y_accel'][i-index-1] * (row['time_stamp'] - new_df['time_stamp'][i-index-1]) + new_df['y_delta_vel'][i-index - 1]
            row['z_delta_vel'] = new_df['z_accel'][i-index-1] * (row['time_stamp'] - new_df['time_stamp'][i-index-1]) + new_df['z_delta_vel'][i-index - 1]
        else:
            row['x_delta_vel'] = 0
            row['y_delta_vel'] = 0
            row['z_delta_vel'] = 0
        
        if i != index:
            row['x_delta'] = new_df['x_delta_vel'][i-index-1] * (row['time_stamp'] - new_df['time_stamp'][i-index-1]) + new_df['x_delta'][i-index - 1]
            row['y_delta'] = new_df['y_delta_vel'][i-index-1] * (row['time_stamp'] - new_df['time_stamp'][i-index-1]) + new_df['y_delta'][i-index - 1]
            row['z_delta'] = new_df['z_delta_vel'][i-index-1] * (row['time_stamp'] - new_df['time_stamp'][i-index-1]) + new_df['z_delta'][i-index - 1]
        else:
            row['x_delta'] = 0
            row['y_delta'] = 0
            row['z_delta'] = 0

        (row['qx'], row['qy'], row['qz'], row['qw']) = R.from_matrix(\
                            rot_mat_from_new_reference_to_old_reference.T\
                                @(imu_orientation_interp(frame_to_time_df['#timestamp [ns]'][i]).as_matrix())\
                            )\
                            .as_quat()\

        row = pd.DataFrame(row)
        new_df = pd.concat((new_df, row), ignore_index=True)
        # print(new_df) 
    #     print(i, '/', len(frame_to_time_df))
    print(write_path + '/reference_frame_index_'+str(index))
    os.makedirs(write_path + '/reference_frame_index_'+str(index), exist_ok = True)
    new_df.to_csv(write_path + '/reference_frame_index_'+str(index) +'/data.csv')

    return rot_mat_from_new_reference_to_old_reference

if __name__ == '__main__':
    get_info_from_frame(0, '/home/tau/Desktop/monocular_data/dataset-corridor1_512_16', '/home/tau/Desktop/monocular_data/dataset-corridor1_512_16')
