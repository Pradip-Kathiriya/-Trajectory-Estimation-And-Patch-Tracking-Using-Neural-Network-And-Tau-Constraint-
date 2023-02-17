import string
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R
import pandas as pd



def get_interpolation(path:string):
    
    position_file = pd.read_csv(path + '/mav0/imu0/data.csv')
    time_diff_secs = np.array(position_file['#timestamp [ns]'])[1:] - np.array(position_file['#timestamp [ns]'])[0:-1]
    time_diff_secs = np.array(time_diff_secs, dtype=float)
    time_diff_secs /= 10.0**9

    def quat_mult(q, p):
        r = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r[0] = q[0]*p[0] - q[1]*p[1] - q[2]*p[2] - q[3]*p[3]
        r[1] = q[0]*p[1] + q[1]*p[0] + q[2]*p[3] - q[3]*p[2]
        r[2] = q[0]*p[2] - q[1]*p[3] + q[2]*p[0] + q[3]*p[1]
        r[3] = q[0]*p[3] + q[1]*p[2] - q[2]*p[1] + q[3]*p[0]
        return r
    # Scalar first
    # Forward eular, could do trapezoidal
    def integrate_quaternion(q, gyr, dt):
        p = np.array([0.0, gyr[0], gyr[1], gyr[2]], dtype=np.float32)
        dot_q = 0.5 * quat_mult(q, p)
        q_unpacked = q + dt * dot_q
        return q_unpacked
    
    quad_list = [[1.0, 0.0, 0.0, 0.0]]

    i = 0
    for delta_time in time_diff_secs:
        quad_list.append(integrate_quaternion(\
            quad_list[-1], \
            (position_file['w_RS_S_x [rad s^-1]'][i], position_file['w_RS_S_y [rad s^-1]'][i], position_file['w_RS_S_z [rad s^-1]'][i]), \
            delta_time))
        i += 1
    
    quad_array = np.array(quad_list)
    quad_array = np.stack((quad_array[:, 1], quad_array[:, 2], quad_array[:, 3], quad_array[:, 0]), axis=1)

    # print(quad_list)
    rotations = R.from_quat(quad_array)
    quaternion_interp = Slerp(position_file['#timestamp [ns]'], rotations)
    return quaternion_interp

if __name__ == '__main__':
    
    get_interpolation('/home/tau/Desktop/monocular_data/dataset-corridor4_512_16')
    
    

    




