import json

import MinutiaeGrid as mg

# # li = []
# mn = Minutiae()
# minutiae_points = {}
# train_folder_path = "../Data/Train/"
# for folder in os.listdir(train_folder_path):
#     folder_dict = {}
#     for im in os.listdir(train_folder_path + folder):
#         minutiae_list = mn.compute_minutiae(train_folder_path + folder + "\\" + im)
#         folder_dict[im] = minutiae_list
#     minutiae_points[folder] = folder_dict
#
#
# def print_nested_dictionary(dictionary, indent=0):
#     for key, value in dictionary.items():
#         print('\t' * indent + str(key) + ':', end=' ')
#         if isinstance(value, dict):
#             print()
#             print_nested_dictionary(value, indent + 1)
#         else:
#             print(value)
#
#
# print_nested_dictionary(minutiae_points)
#
#
# def create_dataset(dictionary):
#     dataset = tf.data.Dataset.from_tensor_slices(dictionary)
#     dataset = dataset.map(lambda image_id, minutiae_points: (minutiae_points, image_id))
#     return dataset
#
#
# print(create_dataset(minutiae_points))


# Extract minutiae points and labels for train images
train_folder = '../Data/Train/'

minutiae_array, labels = mg.createLabelsMinutiaeArray(train_folder)

minutiae_list = [arr.tolist() for arr in minutiae_array]

minutiae_file_path = '../Data/minutiae_array.json'
labels_file_path = '../Data/labels.json'

mFile = open(minutiae_file_path, 'w')
json.dump(minutiae_list, mFile)
mFile.close()

lFile = open(labels_file_path, 'w')
json.dump(labels, lFile)
lFile.close()

# Extract minutiae points and labels for test images
test_folder = '../Data/Test/'

minutiae_array_test, labels_test = mg.createLabelsMinutiaeArray(test_folder)

minutiae_list_test = [arr.tolist() for arr in minutiae_array_test]

minutiae_file_path_test = '../Data/minutiae_array_test.json'
labels_file_path_test = '../Data/labels_test.json'

mFile_test = open(minutiae_file_path_test, 'w')
json.dump(minutiae_list_test, mFile_test)
mFile_test.close()

lFile_test = open(labels_file_path_test, 'w')
json.dump(labels_test, lFile_test)
lFile_test.close()

