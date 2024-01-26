import os

import numpy as np

import Minutiae as mn


# minutiae_points = {}
# # for folder in os.listdir(train_folder_path):
# #     folder_dict = {}
# #     for im in os.listdir(train_folder_path + folder):
# #         minutiae_list = mn.compute_minutiae(train_folder_path + folder + "\\" + im)
# #         folder_dict[im] = minutiae_list
# #         print(minutiae_list)
# #     minutiae_points[folder] = folder_dict
#

def createLabelsMinutiaeArray(train_folder_path):

    labels = []
    minutiae_array = []
    img_dim = (388, 388)
    for folder in os.listdir(train_folder_path):
        for im in os.listdir(train_folder_path + folder):
            im_matrix = np.zeros(img_dim)
            labels.append(folder)
            minutiae_list = mn.compute_minutiae(train_folder_path + folder + "\\" + im)
            for point in minutiae_list:
                im_matrix[point[0]][point[1]] = 1
            minutiae_array.append(im_matrix)
    return minutiae_array, labels


