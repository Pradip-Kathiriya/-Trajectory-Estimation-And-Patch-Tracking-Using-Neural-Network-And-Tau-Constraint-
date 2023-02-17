import os
import cv2 as cv
import numpy as np
import glob

import os
from collections import defaultdict


import numpy as np


import torch

from tqdm import tqdm



def undistort_images(data_path):
    os.makedirs(data_path + '/processed_images/', exist_ok=True)
    contents = open(data_path + '/dso/cam0/camera.txt', "r").read().split(' ')
    contents[8] = contents[8].split('\n')[0]
    camera_params = contents[1:9]
    camera_params = [float(val) for val in camera_params]
    distCoeff = np.array([camera_params[-4], camera_params[-3], camera_params[-2], camera_params[-1]])

    images = list(glob.glob(data_path + '/dso/cam0/images/*.png'))    
    def sort_key(image_name):
        return int(image_name.split('.')[0].split('/')[-1])
    images.sort(key=sort_key)

    for i, image in tqdm(enumerate(images), total=len(images)):
        image = cv.imread(image, 0)
        height,width = image.shape
        camera_matrix = np.array([[width*camera_params[0] , 0 , width*camera_params[2]-0.5] , [0 ,height*camera_params[1], height*camera_params[3]-0.5] , [0,0,1] ])
        size_factor = 1
        K_new = np.copy(camera_matrix)
        K_new[0, 2] *= size_factor
        K_new[1, 2] *= size_factor
        undistorted = cv.fisheye.undistortImage(image, camera_matrix, distCoeff,Knew=K_new,new_size=(size_factor*width, size_factor*height))

        cv.imshow('distorted',image)
        cv.imshow('undistorted', undistorted)
        cv.waitKey(1)

        img_name = ('0' * (5-len(str(i)))) + str(i) + '.png'
        cv.imwrite(os.path.join(data_path, 'processed_images', 'undistorted_images', img_name), undistorted)

def run_yolo(data_path):
    print(data_path)

    print('\nStarting yolo\n')
    os.system('python3 detect2.py \
               --weights object_detector_weights/yolov7.pt \
               --source ' + os.path.join(data_path, 'processed_images', 'undistorted_images'))
    print('\nDone with yolo\n')
    
    
def generate_data(tensor_path):
    
    tensors = list(glob.glob(tensor_path))
    
    def sort_key(tensor_name):
        return int(tensor_name.split('.')[0].split('/')[-1])
    tensors.sort(key=sort_key)
    
    sequence_dict = defaultdict(list) #Dict to store the sequence
    center_id = {} 
    center_arr = np.array(())
    sequence_num = 0
    center_counter = {}
    first_object = True

    for i ,tensor in enumerate(tensors):

        # if i == 100:
        #     break
        
        det = torch.load(tensor)
        tensor_id = int(tensor.split('.')[0].split('/')[-1])
        # print(tensor_id)

        for *xyxy, conf, cls in reversed(det):
            
            if i == 0:
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                center = np.array(((c1[0] + c2[0])/2, (c1[1] + c2[1])/2)).reshape(1,2)
                sequence_dict[sequence_num].append([tensor_id, (c1, c2)])
                center_id[tuple((center[0][0], center[0][1]))] = sequence_num
                if first_object : 
                    center_arr = np.array((center[0][0], center[0][1])).reshape(1,2)
                if not first_object:
                    center_arr = np.vstack((center_arr, [center[0][0], center[0][1]]))
                center_counter[tuple((center[0][0], center[0][1]))] = 0
                sequence_num += 1
                first_object = False
            
            else:
                
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                center = np.array(((c1[0] + c2[0])/2, (c1[1] + c2[1])/2)).reshape(1,2)
                
                dist_from_center = np.linalg.norm((np.array(center_arr) - np.array(center)), axis=1)
                closest_cent_idx = np.argwhere(dist_from_center < 10)
                if len(closest_cent_idx) > 1:
                    continue
                
                if len(closest_cent_idx) == 1:
                    closest_cent = center_arr[closest_cent_idx].reshape(1,2)
                    seq_num = center_id[tuple((closest_cent[0][0], closest_cent[0][1]))]
                    
                    center_arr = np.vstack((center_arr, [center[0][0], center[0][1]]))
                    center_arr = np.delete(center_arr,closest_cent_idx,0)

                    del center_id[tuple((closest_cent[0][0], closest_cent[0][1]))]
                    center_id[tuple((center[0][0], center[0][1]))] = seq_num

                    sequence_dict[seq_num].append([tensor_id,(c1, c2)])
                    del center_counter[tuple((closest_cent[0][0], closest_cent[0][1]))]
                    center_counter[tuple((center[0][0], center[0][1]))] = 0
                    
                if len(closest_cent_idx) == 0:
                    
                    sequence_dict[sequence_num].append([tensor_id,(c1,c2)])
                    center_id[tuple((center[0][0], center[0][1]))] = sequence_num
                    center_arr = np.vstack((center_arr, [center[0][0], center[0][1]]))
                    sequence_num += 1
                    center_counter[tuple((center[0][0], center[0][1]))] = 0
        
        for key in center_counter:
            center_counter[key] = 1 + center_counter[key]
        
        out_of_the_frame_key = []
        
        for key in center_counter:
            if center_counter[key] == 10:
                out_of_the_frame_key.append(key)

        # print(out_of_the_frame_key)
        for key in out_of_the_frame_key:
            del center_counter[key]
            del center_id[key] 
            center_to_be_removed = np.array((key)).reshape(1,2)
            dist_center_to_be_removed = np.linalg.norm((center_arr - center_to_be_removed), axis=1)
            center_to_be_removed_idx = np.where(dist_center_to_be_removed<2)
            center_arr = np.delete(center_arr, center_to_be_removed_idx, 0)
        # print(center_arr)
    return sequence_dict

if __name__ == '__main__':
    data_path = 'data/monocular_data/dataset-corridor1_512_16'
    tensor_path = 'data/det_tensor/*.pt'
    # undistort_images(data_path)
    # run_yolo(data_path)
    sequence_dict = generate_data(tensor_path)

