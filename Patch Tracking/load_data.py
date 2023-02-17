
from generate_dataset import generate_data
import torch
import cv2 as cv
import os
import numpy as np


class data_set(torch.utils.data.Dataset):
    
    
    def __init__(self, sequence_list):
        data_path = 'data/monocular_data/dataset-corridor1_512_16'
        self.data_list = []
        self.label_list = []
        is_first_image = True
        
        
        for _, val in sequence_list.items():
    
            data_point = []
            label_point = []
            is_first_image = True
            if len(val) < 5:
                continue
        
            else:
                for image_and_bouding_box in val:
                    # print(image_and_bouding_box[1][0][0])
                    if is_first_image:
                        is_first_image = False
                        img_name = ('0' * (5-len(str(image_and_bouding_box[0])))) + str(image_and_bouding_box[0]) + '.png'
                        first_image = cv.imread(os.path.join(data_path, 'processed_images', 'undistorted_images', img_name))
                        first_bound_box = np.array([image_and_bouding_box[1][0][0], image_and_bouding_box[1][0][1], image_and_bouding_box[1][1][0],image_and_bouding_box[1][1][1]])
                        data_1 = np.hstack((first_image,first_image))
                        label_1 = np.vstack((first_bound_box,first_bound_box))
                        data_point.append(data_1)
                        label_point.append(label_1)
                        
                    else:
                        img_name = ('0' * (5-len(str(image_and_bouding_box[0])))) + str(image_and_bouding_box[0]) + '.png'
                        next_image = cv.imread(os.path.join(data_path, 'processed_images', 'undistorted_images', img_name))
                        # next_image = cv.imread(os.path.join(data_path, 'processed_images', 'undistorted_images', str(image_and_bouding_box[0]),'.png'))
                        next_bound_box = np.array([image_and_bouding_box[1][0][0], image_and_bouding_box[1][0][1], image_and_bouding_box[1][1][0],image_and_bouding_box[1][1][1]])
                        data_2 = np.hstack((first_image,next_image))
                        label_2 = np.vstack((first_bound_box,next_bound_box))
                        data_point.append(data_2)
                        label_point.append(label_2)
                    
                    self.data_list.append(data_point)
                    self.label_list.append(label_point)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index], self.label_list[index]


if __name__ == '__main__':
    tensor_path = 'data/det_tensor/*.pt'
    sequence_dict = generate_data(tensor_path)
    data_path = 'data/monocular_data/dataset-corridor1_512_16'
    # data_list = []
    # label_list = []
    # for keys, val in sequence_dict.items():

    #     data_point = []
    #     label_point = []
    #     is_first_image = True
    #     if len(val) < 5:
    #         continue
    
    #     else:
    #         for image_and_bouding_box in val:
    #             # print(image_and_bouding_box[1][0][0])
    #             if is_first_image:
    #                 is_first_image = False
    #                 img_name = ('0' * (5-len(str(image_and_bouding_box[0])))) + str(image_and_bouding_box[0]) + '.png'
    #                 first_image = cv.imread(os.path.join(data_path, 'processed_images', 'undistorted_images', img_name))
    #                 first_bound_box = np.array([image_and_bouding_box[1][0][0], image_and_bouding_box[1][0][1], image_and_bouding_box[1][1][0],image_and_bouding_box[1][1][1]])
    #                 data_1 = np.hstack((first_image,first_image))
    #                 label_1 = np.vstack((first_bound_box,first_bound_box))
    #                 data_point.append(data_1)
    #                 label_point.append(label_1)
                    
    #             else:
    #                 img_name = ('0' * (5-len(str(image_and_bouding_box[0])))) + str(image_and_bouding_box[0]) + '.png'
    #                 next_image = cv.imread(os.path.join(data_path, 'processed_images', 'undistorted_images', img_name))
    #                 # next_image = cv.imread(os.path.join(data_path, 'processed_images', 'undistorted_images', str(image_and_bouding_box[0]),'.png'))
    #                 next_bound_box = np.array([image_and_bouding_box[1][0][0], image_and_bouding_box[1][0][1], image_and_bouding_box[1][1][0],image_and_bouding_box[1][1][1]])
    #                 data_2 = np.hstack((first_image,next_image))
    #                 label_2 = np.vstack((first_bound_box,next_bound_box))
    #                 data_point.append(data_2)
    #                 label_point.append(label_2)
            
    #         data_list.append(data_point)
    #         label_list.append(label_point)
    #     break
    
    # cv.imshow('image_1', data_list[0][0])
    # cv.waitKey(0)
    
    # print(data_list[0][0].shape)

     